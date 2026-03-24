from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, relative_path: str) -> Any:
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


data_mod = load_module("cs336_local_data", "cs336_basics/data.py")
loss_mod = load_module("cs336_local_loss", "cs336_basics/loss.py")
model_mod = load_module("cs336_local_model", "cs336_basics/model.py")

get_batch = data_mod.get_batch
cross_entropy = loss_mod.cross_entropy
TransformerLM = model_mod.TransformerLM


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CS336 learning-rate tuning experiment with SGD.")
    parser.add_argument("--train-bin", default=str(ROOT / "data" / "tinystories_train.bin"))
    parser.add_argument("--token-dtype", default="uint16", choices=("uint16", "uint32", "uint64"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e1, 1e2, 1e3],
        help="Learning rates to test.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "learning_rate_tuning" / "outputs"),
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )


def load_memmap(path: str, token_dtype: str) -> np.memmap:
    return np.memmap(path, dtype=np.dtype(token_dtype), mode="r")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = output_dir / "lr_tuning_steps.jsonl"
    summary_path = output_dir / "lr_tuning_summary.json"

    train_data = load_memmap(args.train_bin, args.token_dtype)
    if len(train_data) <= args.context_length:
        raise ValueError("Training data is too short for the chosen context length.")

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for lr in args.learning_rates:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = build_model(args)
        model.train()
        optimizer = SGD(model.parameters(), lr=lr)

        lr_records: list[dict[str, Any]] = []
        print(f"Running learning rate {lr:g} on device={args.device}")

        for step in range(args.steps):
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits, y)

            loss_value = float(loss.item())
            record = {
                "learning_rate": lr,
                "step": step,
                "loss": loss_value,
                "is_finite": bool(torch.isfinite(loss).item()),
            }
            lr_records.append(record)
            all_records.append(record)
            print(f"  step={step:02d} loss={loss_value:.6f}")

            if not torch.isfinite(loss):
                print("  Encountered non-finite loss; stopping this run early.")
                break

            loss.backward()
            optimizer.step()

        summaries.append(
            {
                "learning_rate": lr,
                "num_recorded_steps": len(lr_records),
                "final_loss": lr_records[-1]["loss"] if lr_records else None,
                "min_loss": min((r["loss"] for r in lr_records), default=None),
                "max_loss": max((r["loss"] for r in lr_records), default=None),
                "all_finite": all(r["is_finite"] for r in lr_records),
            }
        )

    with steps_path.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")

    summary = {
        "train_bin": args.train_bin,
        "device": args.device,
        "steps_per_run": args.steps,
        "learning_rates": args.learning_rates,
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Wrote step records to {steps_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
