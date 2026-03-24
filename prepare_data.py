import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics import Tokenizer


SPECIAL_TOKENS = ["<|endoftext|>"]


def choose_token_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def load_tokenizer(
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
) -> tuple[Tokenizer, np.dtype, int]:
    with vocab_path.open("rb") as f:
        vocab = pickle.load(f)
    with merges_path.open("rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    vocab_size = len(vocab)
    token_dtype = choose_token_dtype(vocab_size)
    return tokenizer, token_dtype, vocab_size


def metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def partial_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".partial")


def completed_output_matches(
    input_path: Path,
    output_path: Path,
    token_dtype: np.dtype,
    vocab_size: int,
) -> bool:
    meta_path = metadata_path(output_path)
    if not output_path.exists() or not meta_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return (
        meta.get("input_path") == str(input_path)
        and meta.get("input_size_bytes") == input_path.stat().st_size
        and meta.get("dtype") == str(token_dtype)
        and meta.get("vocab_size") == vocab_size
        and meta.get("output_size_bytes") == output_path.stat().st_size
    )


def encode_text_to_bin(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    token_dtype: np.dtype,
    vocab_size: int,
    chunk_size: int,
    report_every_bytes: int,
    overwrite: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = partial_path(output_path)
    meta_output = metadata_path(output_path)

    if completed_output_matches(input_path, output_path, token_dtype, vocab_size) and not overwrite:
        existing_meta = json.loads(meta_output.read_text(encoding="utf-8"))
        print(f"Skipping completed file: {output_path}")
        print(f"Existing tokens: {existing_meta['num_tokens']}")
        return int(existing_meta["num_tokens"])

    if overwrite:
        output_path.unlink(missing_ok=True)
        meta_output.unlink(missing_ok=True)
    temp_output.unlink(missing_ok=True)

    total_input_bytes = input_path.stat().st_size
    processed_bytes = 0
    last_report_bytes = 0
    num_tokens = 0
    token_buffer: list[int] = []
    start_time = time.time()

    with input_path.open("r", encoding="utf-8") as in_f, temp_output.open("wb") as out_f:
        for line_idx, line in enumerate(in_f, start=1):
            processed_bytes += len(line.encode("utf-8"))
            token_buffer.extend(tokenizer.encode(line))

            while len(token_buffer) >= chunk_size:
                chunk = np.asarray(token_buffer[:chunk_size], dtype=token_dtype)
                chunk.tofile(out_f)
                num_tokens += int(chunk.size)
                del token_buffer[:chunk_size]

            if processed_bytes - last_report_bytes >= report_every_bytes:
                elapsed = max(time.time() - start_time, 1e-6)
                pct = 100.0 * processed_bytes / max(total_input_bytes, 1)
                rate = processed_bytes / elapsed
                print(
                    f"[{input_path.name}] {pct:6.2f}% "
                    f"({format_bytes(processed_bytes)}/{format_bytes(total_input_bytes)}) "
                    f"tokens={num_tokens:,} "
                    f"rate={format_bytes(int(rate))}/s "
                    f"line={line_idx:,}",
                    flush=True,
                )
                last_report_bytes = processed_bytes

        if token_buffer:
            chunk = np.asarray(token_buffer, dtype=token_dtype)
            chunk.tofile(out_f)
            num_tokens += int(chunk.size)

    temp_output.replace(output_path)

    meta = {
        "input_path": str(input_path),
        "input_size_bytes": total_input_bytes,
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "dtype": str(token_dtype),
        "vocab_size": vocab_size,
        "num_tokens": num_tokens,
        "special_tokens": SPECIAL_TOKENS,
    }
    meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print(
        f"Completed {input_path.name}: tokens={num_tokens:,}, "
        f"output={output_path.name}, elapsed={elapsed / 60:.2f} min",
        flush=True,
    )
    return num_tokens


def default_jobs(data_dir: Path, split: str) -> list[tuple[Path, Path]]:
    jobs = {
        "train": (
            data_dir / "TinyStoriesV2-GPT4-train.txt",
            data_dir / "tinystories_train.bin",
        ),
        "valid": (
            data_dir / "TinyStoriesV2-GPT4-valid.txt",
            data_dir / "tinystories_val.bin",
        ),
    }
    if split == "all":
        return [jobs["train"], jobs["valid"]]
    return [jobs[split]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize text datasets into binary token-id files for memmap training."
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("checkpoints/tinystories_vocab.pkl"),
        help="Path to the pickled vocab file.",
    )
    parser.add_argument(
        "--merges",
        type=Path,
        default=Path("checkpoints/tinystories_merges.pkl"),
        help="Path to the pickled merges file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw text datasets.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "all"),
        default="all",
        help="Which TinyStories split to encode.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional single input text file to encode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional single output .bin file. Must be provided together with --input.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="How many token ids to buffer before each write.",
    )
    parser.add_argument(
        "--report-every-mb",
        type=int,
        default=128,
        help="Progress reporting interval in input megabytes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force regeneration even if a completed output already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (args.input is None) != (args.output is None):
        raise ValueError("--input and --output must be provided together.")

    tokenizer, token_dtype, vocab_size = load_tokenizer(args.vocab, args.merges, SPECIAL_TOKENS)
    report_every_bytes = max(args.report_every_mb, 1) * 1024 * 1024

    if args.input is not None:
        jobs = [(args.input, args.output)]
    else:
        jobs = default_jobs(args.data_dir, args.split)

    print(f"Using tokenizer from {args.vocab} and {args.merges}")
    print(f"Vocab size: {vocab_size}")
    print(f"Writing token ids as {token_dtype}")
    print(f"Chunk size: {args.chunk_size:,} tokens")
    print(f"Report interval: {args.report_every_mb} MB")

    for input_path, output_path in jobs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        print(f"Encoding {input_path} -> {output_path}", flush=True)
        encode_text_to_bin(
            tokenizer=tokenizer,
            input_path=input_path,
            output_path=output_path,
            token_dtype=token_dtype,
            vocab_size=vocab_size,
            chunk_size=args.chunk_size,
            report_every_bytes=report_every_bytes,
            overwrite=args.overwrite,
        )

    print("All requested datasets are ready for np.memmap loading in train.py.", flush=True)


if __name__ == "__main__":
    main()
