from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from cs336_basics.tokenizer_optimized import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on an OpenWebText text file.")
    parser.add_argument("--input-path", default="data/owt_train.txt", help="Path to the OpenWebText training text.")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Target vocabulary size.")
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to add to the vocabulary. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/openwebtext_bpe",
        help="Directory where vocab, merges, and summary files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Starting OpenWebText BPE training on {input_path}")
    print(f"Target vocab size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")

    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    elapsed_seconds = time.perf_counter() - start_time

    longest_token = max(vocab.values(), key=len)
    try:
        longest_token_text = longest_token.decode("utf-8")
    except UnicodeDecodeError:
        longest_token_text = None

    vocab_path = output_dir / "owt_vocab.pkl"
    merges_path = output_dir / "owt_merges.pkl"
    summary_path = output_dir / "owt_bpe_summary.json"

    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

    summary = {
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "requested_vocab_size": args.vocab_size,
        "special_tokens": args.special_tokens,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "longest_token_len": len(longest_token),
        "longest_token_utf8": longest_token_text,
        "longest_token_bytes_repr": repr(longest_token),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"Training complete in {elapsed_seconds / 60.0:.2f} minutes")
    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Longest token bytes: {repr(longest_token)}")
    if longest_token_text is not None:
        print(f"Longest token text: {longest_token_text!r}")


if __name__ == "__main__":
    main()
