from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lightweight sampled OpenWebText shard by streaming lines.")
    parser.add_argument("--input-path", default="data/owt_train.txt", help="Path to the full OpenWebText text file.")
    parser.add_argument(
        "--output-path",
        default="data/openwebtext_sampled_2GB.txt",
        help="Path to the sampled output text file.",
    )
    parser.add_argument(
        "--keep-prob",
        type=float,
        default=0.2,
        help="Probability of keeping each input line.",
    )
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=2.0,
        help="Stop once the sampled file reaches about this size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-every-mb",
        type=int,
        default=256,
        help="Progress report interval for written output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not (0.0 < args.keep_prob <= 1.0):
        raise ValueError("--keep-prob must be in (0, 1].")

    target_size_bytes = int(args.target_size_gb * (1024**3))
    report_every_bytes = max(args.report_every_mb, 1) * 1024 * 1024
    rng = random.Random(args.seed)

    total_input_lines = 0
    kept_lines = 0
    output_bytes = 0
    last_report_bytes = 0
    start_time = time.perf_counter()

    print(f"Sampling from {input_path}")
    print(f"Writing sampled shard to {output_path}")
    print(f"keep_prob={args.keep_prob}, target_size={format_bytes(target_size_bytes)}, seed={args.seed}")

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            total_input_lines += 1
            if rng.random() >= args.keep_prob:
                continue

            out_f.write(line)
            kept_lines += 1
            output_bytes += len(line.encode("utf-8"))

            if output_bytes - last_report_bytes >= report_every_bytes:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                rate = output_bytes / elapsed
                pct = min(100.0, 100.0 * output_bytes / max(target_size_bytes, 1))
                print(
                    f"[sample] {pct:6.2f}% "
                    f"written={format_bytes(output_bytes)} "
                    f"kept_lines={kept_lines:,} "
                    f"seen_lines={total_input_lines:,} "
                    f"rate={format_bytes(int(rate))}/s",
                    flush=True,
                )
                last_report_bytes = output_bytes

            if output_bytes >= target_size_bytes:
                break

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "keep_prob": args.keep_prob,
        "target_size_bytes": target_size_bytes,
        "actual_size_bytes": output_bytes,
        "seed": args.seed,
        "seen_lines": total_input_lines,
        "kept_lines": kept_lines,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
    }

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Sampling complete in {elapsed_seconds / 60.0:.2f} minutes")
    print(f"Sampled file size: {format_bytes(output_bytes)}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
