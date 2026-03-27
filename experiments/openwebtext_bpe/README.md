# OpenWebText BPE Experiments

This folder contains two practical workflows:

1. `run_openwebtext_bpe.py`
   A direct full-file BPE trainer. Use this only if you have enough RAM for the
   entire input text plus the BPE bookkeeping structures.
2. `make_openwebtext_sample.py` + `run_openwebtext_bpe_chunked.py`
   A safer local workflow for a 32 GB machine:
   first create a sampled shard, then train BPE on that shard with chunked
   pretokenization.

## Full-file workflow

```bash
python experiments/openwebtext_bpe/run_openwebtext_bpe.py \
  --input-path data/owt_train.txt \
  --vocab-size 32000 \
  --output-dir outputs/openwebtext_bpe
```

Generated outputs:

- `outputs/openwebtext_bpe/owt_vocab.pkl`
- `outputs/openwebtext_bpe/owt_merges.pkl`
- `outputs/openwebtext_bpe/owt_bpe_summary.json`

## Sampled + chunked local workflow

### Step 1: stream-sample a local shard

```bash
python experiments/openwebtext_bpe/make_openwebtext_sample.py \
  --input-path data/owt_train.txt \
  --output-path data/openwebtext_sampled_2GB.txt \
  --keep-prob 0.2 \
  --target-size-gb 2.0
```

This reads the full `owt_train.txt` file line by line and keeps roughly 20% of
lines until the sampled output reaches about 2 GB.

### Step 2: train BPE on the sampled file with chunked pretokenization

```bash
python experiments/openwebtext_bpe/run_openwebtext_bpe_chunked.py \
  --input-path data/openwebtext_sampled_2GB.txt \
  --vocab-size 32000 \
  --chunk-size-mb 128 \
  --output-dir outputs/openwebtext_bpe_chunked
```

Generated outputs:

- `outputs/openwebtext_bpe_chunked/owt_chunked_vocab.pkl`
- `outputs/openwebtext_bpe_chunked/owt_chunked_merges.pkl`
- `outputs/openwebtext_bpe_chunked/owt_chunked_bpe_summary.json`

### Recommended stable launcher on Windows

For long local runs, prefer the PowerShell launcher below. It resolves absolute
paths, validates the Python environment and write permissions, stores a run
manifest, and captures the full native process log to disk.

```powershell
powershell -ExecutionPolicy Bypass -File experiments\openwebtext_bpe\run_openwebtext_bpe_chunked_stable.ps1 `
  -InputPath data\openwebtext_sampled_2_5GB.txt `
  -VocabSize 32000 `
  -ChunkSizeMb 128 `
  -OutputDir outputs\openwebtext_bpe_chunked
```

To validate the environment without starting training:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\openwebtext_bpe\run_openwebtext_bpe_chunked_stable.ps1 `
  -InputPath data\openwebtext_sampled_2_5GB.txt `
  -OutputDir outputs\openwebtext_bpe_chunked `
  -ValidateOnly
```

## Important note

The chunked workflow only chunks the raw-text pretokenization phase. It avoids
duplicating the entire input text in memory, but the global `word_freq`,
`pair_counts`, `pair_to_words`, and merge structures still live in memory. This
is much safer for a sampled 2 GB shard on a 32 GB machine, but it is not a
guarantee that full 11.9 GB OpenWebText will fit.
