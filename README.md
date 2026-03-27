# CS336 Assignment 1 Submission Notes

本仓库是我完成 Stanford CS336 Assignment 1（Basics）的作业提交仓库，内容包括：

- 代码实现
- 书面题与实验题答案
- 用于生成图表和实验结果的辅助脚本
- 少量需要保留的最终图片产物

## 建议阅读顺序

1. 书面题与最终整理版交付物：
   - [deliverable.md](./deliverable.md)
2. 主文档（包含更完整的中文整理、实验图和分析）：
   - [cs336_spring2025_assignment1_basics_zh_v2.md](./cs336_spring2025_assignment1_basics_zh_v2.md)
3. 原版作业 handout：
   - [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## 代码实现位置

主要实现集中在 `cs336_basics/`：

- [cs336_basics/tokenizer_optimized.py](./cs336_basics/tokenizer_optimized.py)
  - BPE 训练
  - `Tokenizer`
- [cs336_basics/model.py](./cs336_basics/model.py)
  - `Linear`
  - `Embedding`
  - `RMSNorm`
  - `SwiGLU`
  - `RotaryPositionalEmbedding`
  - attention / Transformer block / Transformer LM
- [cs336_basics/loss.py](./cs336_basics/loss.py)
  - `cross_entropy`
- [cs336_basics/optimizer.py](./cs336_basics/optimizer.py)
  - `AdamW`
- [cs336_basics/scheduler.py](./cs336_basics/scheduler.py)
  - cosine lr schedule
  - gradient clipping
- [cs336_basics/data.py](./cs336_basics/data.py)
  - data loader
  - checkpointing
- [cs336_basics/train.py](./cs336_basics/train.py)
  - 完整训练循环
- [cs336_basics/generate.py](./cs336_basics/generate.py)
  - temperature / top-p 解码

## 实验与作图脚本

实验相关脚本集中在 `experiments/`：

- [experiments/plot_metrics_comparison.py](./experiments/plot_metrics_comparison.py)
  - 通用学习曲线对比作图
- [experiments/learning_rate_tuning/plot_lr_sweep.py](./experiments/learning_rate_tuning/plot_lr_sweep.py)
  - 学习率 sweep 作图
- [experiments/openwebtext_bpe/](./experiments/openwebtext_bpe/)
  - OpenWebText 采样与 chunked BPE 训练脚本
- [experiments/ablation_plots/](./experiments/ablation_plots/)
  - 部分消融图的专用绘图脚本

## 最终保留的图

最终用于答案展示的图片保存在 `outputs/`：

- `outputs/ablation_plots/`
- `outputs/batch_size_plots/`

这些图已经在主文档和 `deliverable.md` 中被引用。

## 运行方式

环境使用 `uv` 管理。常用命令：

```bash
uv run pytest
uv run python prepare_data.py
uv run python -m cs336_basics.train
uv run python -m cs336_basics.generate
```

## 说明

- 本仓库已经清理掉大部分本地临时记录、备份文档和中间实验垃圾文件。
- 数据集、checkpoint、大型日志等默认不随仓库提交。
- 若需要查看最终作答内容，应优先查看 [deliverable.md](./deliverable.md)。
