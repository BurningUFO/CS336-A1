# OpenWebText BPE On Colab

This folder contains a standalone script for training a byte-level BPE tokenizer
on an OpenWebText text file and saving the resulting artifacts.

Recommended Colab workflow:

```bash
git clone git@github.com:BurningUFO/CS336-A1.git
cd CS336-A1
python -m pip install -e .
```

If you want the Stanford sample OpenWebText files:

```bash
python download_data.py
```

Then run:

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

Important note:

- The current `train_bpe` implementation reads the full input text into memory.
- Full OpenWebText training may require a high-RAM runtime or a smaller shard if
  standard Colab RAM is not enough.

