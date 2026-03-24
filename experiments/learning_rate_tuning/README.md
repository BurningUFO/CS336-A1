# Learning Rate Tuning Experiment

This folder contains a small standalone experiment for CS336 assignment 1 problem `learning_rate_tuning`.

It runs the handout-style SGD optimizer with learning rates `1e1`, `1e2`,and `1e3`, for 10 update steps each, and records the loss after every step.

Recommended command:

```bash
uv run python experiments/learning_rate_tuning/run_sgd_lr_tuning.py
```

Outputs are written to:

- `experiments/learning_rate_tuning/outputs/lr_tuning_steps.jsonl`
- `experiments/learning_rate_tuning/outputs/lr_tuning_summary.json`

