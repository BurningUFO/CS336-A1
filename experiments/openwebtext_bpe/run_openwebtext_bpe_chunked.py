from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

from cs336_basics.tokenizer_optimized import (
    _ReverseLexPair,
    _add_word_to_pair_index,
    _get_word_pair_counts,
    _pop_best_pair,
    _push_pair_heap,
    _remove_word_from_pair_index,
    build_word_freq_from_text,
    merge_word,
)


WordTokens = tuple[bytes, ...]
TokenPair = tuple[bytes, bytes]


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE on a sampled OpenWebText shard with chunked pretokenization.")
    parser.add_argument("--input-path", default="data/openwebtext_sampled_2GB.txt")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to add to the vocabulary. Can be provided multiple times.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=128,
        help="How much text to accumulate per pretokenization chunk.",
    )
    parser.add_argument(
        "--report-every-chunks",
        type=int,
        default=4,
        help="Print a progress report every N chunks.",
    )
    parser.add_argument(
        "--report-every-merges",
        type=int,
        default=1000,
        help="Print a progress report every N merges. Use a very large value to suppress merge-loop logging.",
    )
    parser.add_argument("--output-dir", default="outputs/openwebtext_bpe_chunked")
    return parser.parse_args()


def iter_text_chunks(input_path: Path, chunk_size_bytes: int):
    buffer: list[str] = []
    buffered_bytes = 0
    chunk_index = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_bytes = len(line.encode("utf-8"))
            if buffer and buffered_bytes + line_bytes > chunk_size_bytes:
                chunk_index += 1
                yield chunk_index, "".join(buffer), buffered_bytes
                buffer = []
                buffered_bytes = 0
            buffer.append(line)
            buffered_bytes += line_bytes
        if buffer:
            chunk_index += 1
            yield chunk_index, "".join(buffer), buffered_bytes


def build_word_freq_chunked(
    input_path: Path,
    special_tokens: list[str],
    chunk_size_bytes: int,
    report_every_chunks: int,
) -> tuple[dict[WordTokens, int], dict[str, int | float]]:
    word_freq: Counter[WordTokens] = Counter()
    chunk_count = 0
    processed_bytes = 0
    total_input_bytes = input_path.stat().st_size
    start_time = time.perf_counter()

    for chunk_index, chunk_text, chunk_bytes in iter_text_chunks(input_path, chunk_size_bytes):
        chunk_count = chunk_index
        chunk_word_freq = build_word_freq_from_text(chunk_text, special_tokens)
        word_freq.update(chunk_word_freq)
        processed_bytes += chunk_bytes

        if chunk_index % max(report_every_chunks, 1) == 0:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            rate = processed_bytes / elapsed
            pct = 100.0 * processed_bytes / max(total_input_bytes, 1)
            print(
                f"[chunked-pretoken] chunk={chunk_index:04d} "
                f"processed={pct:6.2f}% "
                f"({format_bytes(processed_bytes)}/{format_bytes(total_input_bytes)}) "
                f"unique_words={len(word_freq):,} "
                f"rate={format_bytes(int(rate))}/s",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - start_time
    return dict(word_freq), {
        "chunk_count": chunk_count,
        "processed_bytes": processed_bytes,
        "elapsed_seconds": elapsed_seconds,
    }


def train_bpe_chunked(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    chunk_size_bytes: int,
    report_every_chunks: int,
    report_every_merges: int,
) -> tuple[dict[int, bytes], list[TokenPair], dict[str, int | float]]:
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    vocab_values = set(vocab.values())
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab_values:
            vocab[next_id] = token_bytes
            vocab_values.add(token_bytes)
            next_id += 1

    print("Starting chunked pretokenization and word frequency aggregation...")
    word_freq, pretoken_stats = build_word_freq_chunked(
        input_path=input_path,
        special_tokens=special_tokens,
        chunk_size_bytes=chunk_size_bytes,
        report_every_chunks=report_every_chunks,
    )
    print(
        f"Pretokenization complete: unique_words={len(word_freq):,}, "
        f"elapsed={pretoken_stats['elapsed_seconds'] / 60.0:.2f} min"
    )

    merges: list[TokenPair] = []
    target_num_merges = vocab_size - len(vocab)
    pair_counts: Counter[TokenPair] = Counter()
    pair_to_words: dict[TokenPair, set[WordTokens]] = defaultdict(set)
    pair_heap: list[tuple[int, _ReverseLexPair, TokenPair]] = []

    merge_setup_start = time.perf_counter()
    for token_seq, freq in word_freq.items():
        word_pair_counts = _get_word_pair_counts(token_seq)
        for pair, count in word_pair_counts.items():
            pair_counts[pair] += count * freq
        _add_word_to_pair_index(pair_to_words, token_seq)
    for pair, count in pair_counts.items():
        _push_pair_heap(pair_heap, pair, count)
    merge_setup_seconds = time.perf_counter() - merge_setup_start

    print("Starting merge loop...")
    merge_loop_start = time.perf_counter()
    for merge_idx in range(target_num_merges):
        best_pair = _pop_best_pair(pair_heap, pair_counts)
        if best_pair is None:
            break

        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1

        affected_words = list(pair_to_words.get(best_pair, ()))
        if not affected_words:
            pair_counts.pop(best_pair, None)
            continue

        for token_seq in affected_words:
            freq = word_freq.pop(token_seq, 0)
            if freq == 0:
                continue

            old_pair_counts = _get_word_pair_counts(token_seq)
            _remove_word_from_pair_index(pair_to_words, token_seq)
            for pair, count in old_pair_counts.items():
                updated_count = pair_counts[pair] - (count * freq)
                if updated_count > 0:
                    pair_counts[pair] = updated_count
                    _push_pair_heap(pair_heap, pair, updated_count)
                else:
                    pair_counts.pop(pair, None)

            new_seq = merge_word(token_seq, best_pair)
            word_freq[new_seq] = word_freq.get(new_seq, 0) + freq

            new_pair_counts = _get_word_pair_counts(new_seq)
            _add_word_to_pair_index(pair_to_words, new_seq)
            for pair, count in new_pair_counts.items():
                updated_count = pair_counts.get(pair, 0) + (count * freq)
                pair_counts[pair] = updated_count
                _push_pair_heap(pair_heap, pair, updated_count)

        if report_every_merges > 0 and (merge_idx + 1) % report_every_merges == 0:
            elapsed = time.perf_counter() - merge_loop_start
            print(
                f"[merge-loop] merges={merge_idx + 1:,}/{target_num_merges:,} "
                f"elapsed={elapsed / 60.0:.2f} min "
                f"current_vocab={len(vocab):,}",
                flush=True,
            )

    merge_loop_seconds = time.perf_counter() - merge_loop_start
    stats = {
        "pretoken_seconds": pretoken_stats["elapsed_seconds"],
        "merge_setup_seconds": merge_setup_seconds,
        "merge_loop_seconds": merge_loop_seconds,
        "chunk_count": pretoken_stats["chunk_count"],
        "processed_bytes": pretoken_stats["processed_bytes"],
    }
    return vocab, merges, stats


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size_bytes = max(args.chunk_size_mb, 1) * 1024 * 1024
    print(f"Starting chunked OpenWebText BPE training on {input_path}")
    print(f"Target vocab size: {args.vocab_size}")
    print(f"Chunk size: {format_bytes(chunk_size_bytes)}")
    print(f"Special tokens: {args.special_tokens}")

    start_time = time.perf_counter()
    vocab, merges, stats = train_bpe_chunked(
        input_path=input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        chunk_size_bytes=chunk_size_bytes,
        report_every_chunks=args.report_every_chunks,
        report_every_merges=args.report_every_merges,
    )
    elapsed_seconds = time.perf_counter() - start_time

    longest_token = max(vocab.values(), key=len)
    try:
        longest_token_text = longest_token.decode("utf-8")
    except UnicodeDecodeError:
        longest_token_text = None

    vocab_path = output_dir / "owt_chunked_vocab.pkl"
    merges_path = output_dir / "owt_chunked_merges.pkl"
    summary_path = output_dir / "owt_chunked_bpe_summary.json"

    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

    summary = {
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": args.special_tokens,
        "chunk_size_bytes": chunk_size_bytes,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "pretoken_seconds": stats["pretoken_seconds"],
        "merge_setup_seconds": stats["merge_setup_seconds"],
        "merge_loop_seconds": stats["merge_loop_seconds"],
        "chunk_count": stats["chunk_count"],
        "processed_bytes": stats["processed_bytes"],
        "longest_token_len": len(longest_token),
        "longest_token_utf8": longest_token_text,
        "longest_token_bytes_repr": repr(longest_token),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"Chunked training complete in {elapsed_seconds / 60.0:.2f} minutes")
    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Longest token bytes: {repr(longest_token)}")
    if longest_token_text is not None:
        print(f"Longest token text: {longest_token_text!r}")


if __name__ == "__main__":
    main()
