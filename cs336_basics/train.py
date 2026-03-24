from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.loss import cross_entropy
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.scheduler import clip_gradient_norm, get_lr_cosine_schedule


'''
 训练过程中你该看什么
  最重要看这几个：

  1. train_loss

  - 整体应该往下走
  - 短期会抖动，正常
  - 长期不下降才值得担心

  2. val_loss

  - 也应该整体下降
  - 它比 train_loss 更重要，因为它反映泛化
  - 如果 train_loss 降而 val_loss 不降甚至升，说明可能开始过拟合或训练设置不合理

  3. val_perplexity

  - 本质上和 val_loss 同步
  - 越低越好
  - 看趋势，不要只盯某一个点

  4. 训练速度

  - 你的 50 step smoke run 已经证明速度没问题
  - 正式训练时主要确认不会突然异常变慢或报错

  5. checkpoint 是否持续生成

  - 比如 step_0200.pt, step_0400.pt
  - 这说明训练过程稳定可恢复

  什么情况说明训练正常
  你可以把下面当作正常信号：

  - 能稳定跑，不报 CUDA/内存/shape 错误
  - train_loss 持续下降
  - val_loss 也持续下降
  - checkpoint 正常保存
  - metrics.jsonl 在持续追加记录

  什么情况说明要停下来看看
  这些是异常信号：

  - loss 出现 nan
  - val_loss 长时间完全不降
  - 训练特别慢到不合理
  - checkpoint 没有生成
  - 显存报错或程序中断
'''

@dataclass
class TrainConfig:
    train_bin: str = "data/tinystories_train.bin"
    val_bin: str = "data/tinystories_val.bin"
    token_dtype: str = "uint16"
    vocab_size: int = 10_000
    context_length: int = 128
    d_model: int = 256
    d_ff: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    rope_theta: float = 10_000.0
    batch_size: int = 32
    max_steps: int = 3000
    eval_interval: int = 100
    eval_batches: int = 20
    save_interval: int = 200
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints/tinystories_gpu_run1"
    resume_from: str | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a small CS336 Transformer LM.")
    parser.add_argument("--train-bin", default=TrainConfig.train_bin)
    parser.add_argument("--val-bin", default=TrainConfig.val_bin)
    parser.add_argument("--token-dtype", default=TrainConfig.token_dtype, choices=("uint16", "uint32", "uint64"))
    parser.add_argument("--vocab-size", type=int, default=TrainConfig.vocab_size)
    parser.add_argument("--context-length", type=int, default=TrainConfig.context_length)
    parser.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    parser.add_argument("--d-ff", type=int, default=TrainConfig.d_ff)
    parser.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=TrainConfig.num_heads)
    parser.add_argument("--rope-theta", type=float, default=TrainConfig.rope_theta)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=TrainConfig.eval_batches)
    parser.add_argument("--save-interval", type=int, default=TrainConfig.save_interval)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=TrainConfig.min_learning_rate)
    parser.add_argument("--warmup-iters", type=int, default=TrainConfig.warmup_iters)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--checkpoint-dir", default=TrainConfig.checkpoint_dir)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def load_memmap(path: str, token_dtype: str) -> np.memmap:
    return np.memmap(path, dtype=np.dtype(token_dtype), mode="r")


def build_model(cfg: TrainConfig) -> TransformerLM:
    return TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=cfg.device,
    )


def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    cfg: TrainConfig,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = get_batch(dataset, cfg.batch_size, cfg.context_length, cfg.device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def write_jsonl(log_path: Path, payload: dict) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "metrics.jsonl"
    config_path = checkpoint_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_data = load_memmap(cfg.train_bin, cfg.token_dtype)
    val_data = load_memmap(cfg.val_bin, cfg.token_dtype)
    if len(train_data) <= cfg.context_length:
        raise ValueError("Training data is too short for the chosen context length.")
    if len(val_data) <= cfg.context_length:
        raise ValueError("Validation data is too short for the chosen context length.")

    model = build_model(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    start_step = 0
    if cfg.resume_from is not None:
        start_step = load_checkpoint(cfg.resume_from, model, optimizer) + 1

    print(f"device={cfg.device} train_tokens={len(train_data)} val_tokens={len(val_data)}")
    print(
        "model="
        f"(layers={cfg.num_layers}, d_model={cfg.d_model}, heads={cfg.num_heads}, d_ff={cfg.d_ff}, "
        f"context={cfg.context_length})"
    )

    wallclock_start = time.perf_counter()
    model.train()

    for step in range(start_step, cfg.max_steps):
        lr = get_lr_cosine_schedule(
            step,
            max_learning_rate=cfg.learning_rate,
            min_learning_rate=cfg.min_learning_rate,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        optimizer.zero_grad()
        logits = model(x)
        train_loss = cross_entropy(logits, y)
        train_loss.backward()
        clip_gradient_norm(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val_loss, val_ppl = estimate_loss(model, val_data, cfg)
            elapsed = time.perf_counter() - wallclock_start
            metrics = {
                "step": step,
                "train_loss": train_loss.item(),
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "learning_rate": lr,
                "elapsed_seconds": elapsed,
            }
            print(
                f"step={step:04d} train_loss={metrics['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} lr={lr:.6f} "
                f"elapsed={elapsed:.1f}s"
            )
            write_jsonl(log_path, metrics)

        if step % cfg.save_interval == 0 or step == cfg.max_steps - 1:
            ckpt_path = checkpoint_dir / f"step_{step:04d}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)


if __name__ == "__main__":
    main()
