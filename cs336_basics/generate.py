from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer_optimized import Tokenizer


CHECKPOINT_RE = re.compile(r"step_(\d+)\.pt$")
SPECIAL_TOKENS = ["<|endoftext|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 TinyStories checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/tinystories_gpu_run1"),
        help="Directory containing config.json, metrics.jsonl, and saved checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional explicit checkpoint path. If omitted, use the best saved checkpoint by val_loss.",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt to continue from.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Maximum number of tokens to generate after the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Restrict sampling to the top-k logits. Use 0 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("checkpoints/tinystories_vocab.pkl"),
        help="Path to the pickled tokenizer vocab.",
    )
    parser.add_argument(
        "--merges",
        type=Path,
        default=Path("checkpoints/tinystories_merges.pkl"),
        help="Path to the pickled tokenizer merges.",
    )
    parser.add_argument(
        "--allow-eot",
        action="store_true",
        help="If set, do not stop generation when <|endoftext|> is sampled.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
    with vocab_path.open("rb") as f:
        vocab = pickle.load(f)
    with merges_path.open("rb") as f:
        merges = pickle.load(f)
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def choose_best_saved_checkpoint(checkpoint_dir: Path) -> tuple[Path, str]:
    metrics_path = checkpoint_dir / "metrics.jsonl"
    records = load_jsonl(metrics_path)

    saved_checkpoints: dict[int, Path] = {}
    for path in checkpoint_dir.glob("step_*.pt"):
        step = checkpoint_step(path)
        if step is not None:
            saved_checkpoints[step] = path

    if not saved_checkpoints:
        raise FileNotFoundError(f"No saved checkpoints found in {checkpoint_dir}")

    best_overall = min(records, key=lambda r: r["val_loss"])
    saved_metric_records = [r for r in records if int(r["step"]) in saved_checkpoints]
    if not saved_metric_records:
        latest_step = max(saved_checkpoints)
        return saved_checkpoints[latest_step], f"No metric-matched checkpoint found, using latest saved step {latest_step}."

    best_saved = min(saved_metric_records, key=lambda r: r["val_loss"])
    best_saved_step = int(best_saved["step"])
    message = (
        f"Best eval overall: step {best_overall['step']} val_loss={best_overall['val_loss']:.4f}. "
        f"Best saved checkpoint: step {best_saved_step} val_loss={best_saved['val_loss']:.4f}."
    )
    return saved_checkpoints[best_saved_step], message


def build_model_from_config(cfg: dict[str, Any], device: str) -> TransformerLM:
    return TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        num_layers=cfg["num_layers"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg["rope_theta"],
        device=device,
    )


def load_checkpoint_weights(model: TransformerLM, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        top_values, _ = torch.topk(logits, top_k)
        cutoff = top_values[-1]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_ids(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    stop_token_id: int | None,
) -> list[int]:
    generated = list(prompt_ids)
    context_length = len(model.layers[0].attn.rope.cos_cached) if model.layers and model.layers[0].attn.rope is not None else None

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if context_length is not None:
                model_input_ids = generated[-context_length:]
            else:
                model_input_ids = generated

            x = torch.tensor([model_input_ids], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            next_token_id = sample_next_token(logits, temperature, top_k)
            generated.append(next_token_id)

            if stop_token_id is not None and next_token_id == stop_token_id:
                break

    return generated


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_dir = args.checkpoint_dir
    config = load_json(checkpoint_dir / "config.json")

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        selection_message = f"Using explicit checkpoint: {checkpoint_path}"
    else:
        checkpoint_path, selection_message = choose_best_saved_checkpoint(checkpoint_dir)

    tokenizer = load_tokenizer(args.vocab, args.merges)
    stop_token_id = None
    if not args.allow_eot:
        stop_token_id = tokenizer.inverse_vocab.get(b"<|endoftext|>")

    model = build_model_from_config(config, args.device)
    load_checkpoint_weights(model, checkpoint_path)
    model.to(args.device)

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens. Provide a non-empty prompt.")

    generated_ids = generate_ids(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        stop_token_id=stop_token_id,
    )
    generated_text = tokenizer.decode(generated_ids)

    print(selection_message)
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Generated token count: {len(generated_ids) - len(prompt_ids)}")
    print()
    print(generated_text)


if __name__ == "__main__":
    main()
