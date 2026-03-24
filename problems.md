# 2 字节对编码（BPE）Tokenizer
## 2.1 Unicode 标准

### Problem (`unicode1`)：理解 Unicode（1 分）

(a) `chr(0)` 返回的 Unicode 字符是什么？  
**Deliverable**：chr(0) 返回的是 Unicode 字符 U+0000，也就是空字符（NUL）。

(b) 该字符的字符串表示（`__repr__()`）与其打印表示有何不同？  
**Deliverable**：它的 __repr__() 会显示成可见的转义形式 '\x00'，而直接打印时通常是不可见字符，看起来像“什么都没输出”。

(c) 当该字符出现在文本中时会发生什么？你可以在 Python 解释器中尝试以下代码，并观察其行为是否符合预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**Deliverable**：当它出现在 Python 字符串里时不会截断字符串，只是作为一个不可见控制字符保留在文本中，所以拼接后的字符串长度会包含它，但打印出来通常只会看到中间像“空了一下”而不是显式字符。


## 2.2 Unicode 编码

### Problem (`unicode2`)：Unicode 编码（3 分）

(a) 为什么训练 Tokenizer 时更倾向于使用 UTF-8 编码字节，而不是 UTF-16 或 UTF-32？对若干输入字符串比较这些编码的输出，可能会有帮助。  
**Deliverable**：，因为它对英文和常见文本通常更省空间，而且是按字节自然编码，不会像 UTF-16/UTF-32 那样产生大量 0x00字节，因而更适合做字节级 tokenizer 训练。另一个优点是 UTF-8 没有字节序问题，兼容性也最好。

(b) 下面这个函数本意是把 UTF-8 字节串解码为 Unicode 字符串，但它是**错误的**。为什么？请给出一个会产生错误输出的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**Deliverable**：例如输入 b'\xe4\xbd\xa0'（即 "你" 的 UTF-8 编码）。这个函数错在它把 UTF-8 按“单个字节”分别解码，但 UTF-8 是变长编码，一个字符可能由多个字节共同表示，像 b'\xe4'、b'\xbd'、b'\xa0' 单独都不是完整字符。

(c) 给出一个**无法解码为任何 Unicode 字符**的两字节序列。  
**Deliverable**：例如 b'\xff\xff'。因为 0xFF 不是任何合法 UTF-8 编码单元的一部分，所以这个两字节序列不可能解码成任何 Unicode 字符。


## 2.5 BPE Tokenizer 训练实验

### Problem (`train_bpe`)：BPE Tokenizer 训练（15 分）

**Deliverable**：编写一个函数，输入训练文本文件路径，输出训练好的**字节级 BPE Tokenizer**。你的训练函数至少应支持以下参数：

- `input_path: str`
- `vocab_size: int`
- `special_tokens: list[str]`

应返回：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`

测试方式：

1. 实现 `[adapters.run_train_bpe]`
2. 运行 `uv run pytest tests/test_train_bpe.py`

### Problem (`train_bpe_tinystories`)：在 TinyStories 上训练 BPE（2 分）

(a) 在 TinyStories 上训练一个字节级 BPE Tokenizer，最大词表大小设为 **10,000**，并将 `<|endoftext|>` 加入词表。  
**资源要求**：不使用 GPU 时 ≤ 30 分钟，内存 ≤ 30 GB。  
**Deliverable**：训练总耗时约 22.2 分钟，峰值内存约 10.8 GB，满足题目要求的资源限制。

训练得到的最长 token 是 ' accomplishment'，长度为 15 字节；这很合理，因为 TinyStories 中会反复出现一些带前导空格的常见英文词片段，BPE 会把这些高频共现的字节序列逐步合并成更长token。

(b) 对代码做 profiling，指出最耗时部分。  
**Deliverable**：profiling 和实现分析表明，最耗时的部分是 BPE 的 merge 主循环，尤其是每轮合并时对 pair 频次的维护、受影响词的更新，以及“当前最佳 pair”的选择。相比之下，预分词也有一定开销，但主要瓶颈仍然在反复执行的大量 merge 更新过程。

### Problem (`train_bpe_expts_owt`)：在 OpenWebText 上训练 BPE（2 分）

(a) 在 OpenWebText 上训练一个字节级 BPE Tokenizer，最大词表大小设为 **32,000**。  
**资源要求**：不使用 GPU 时 ≤ 12 小时，内存 ≤ 100 GB。  
**Deliverable**：1 至 2 句话回答最长 Token 及其合理性。

(b) 比较在 TinyStories 与 OpenWebText 上训练出的 Tokenizer。  
**Deliverable**：1 至 2 句话回答。


## 2.6 BPE Tokenizer：编码与解码

### Problem (`tokenizer`)：实现 Tokenizer（15 分）

**Deliverable**：实现一个 `Tokenizer` 类，支持文本到整数 ID 的编码、整数 ID 到文本的解码，并支持用户提供的 Special Tokens。

推荐接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
def encode(self, text: str) -> list[int]
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
def decode(self, ids: list[int]) -> str
```

测试方式：

1. 实现 `[adapters.get_tokenizer]`
2. 运行 `uv run pytest tests/test_tokenizer.py`


## 2.7 实验

### Problem (`tokenizer_experiments`)：Tokenizer 实验（4 分）

(a) 用 TinyStories 与 OpenWebText 的样本文档，测量对应 Tokenizer 的压缩率（bytes/token）。  
**Deliverable**：1 至 2 句话回答。

(b) 用 TinyStories Tokenizer 去编码 OpenWebText，会发生什么？  
**Deliverable**：用 TinyStories tokenizer 去编码 OpenWebText 时，仍然可以正常编码，不会出现 OOV，因为这是字节级 BPE，任何文本最终都能回退到字节表示。只是由于 TinyStories tokenizer 更偏向儿童故事领域，它在 OpenWebText 上通常压缩率更差，会产生更多 token，因为很多开放域文本中的高频词片段和固定搭配并没有在 TinyStories 上被充分学到。

(c) 估算 Tokenizer 吞吐量，并推算编码 The Pile（825 GB）所需时间。  
**Deliverable**：1 至 2 句话回答。

(d) 将训练集和验证集编码为整数 Token ID 序列，建议保存为 `uint16` NumPy 数组。为什么 `uint16` 合适？  
**Deliverable**：uint16 很合适，因为该 tokenizer 的词表大小是 10,000，远小于 2^16 = 65,536，所以每个 token id 都能被 uint16 完整表示。相比 uint32 或 uint64，uint16 可以显著减少存储空间和 I/O 开销，这对大规模语料编码和后续训练都更有利。

---


# 3 Transformer 语言模型架构
## 3.4 基础组件：Linear 与 Embedding 模块

### Problem (`linear`)：实现 Linear 模块（1 分）

**Deliverable**：实现一个继承自 `torch.nn.Module` 的 `Linear` 类。推荐接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
def forward(self, x: torch.Tensor) -> torch.Tensor
```

要求：

- 继承 `nn.Module`
- 调用父类构造函数
- 参数以 `W` 的形式存储，而不是 `W.T`
- 使用 `nn.Parameter`
- **不要**使用 `nn.Linear` 或 `nn.functional.linear`

测试方式：

1. 实现 `[adapters.run_linear]`
2. 运行 `uv run pytest -k test_linear`

### Problem (`embedding`)：实现 Embedding 模块（1 分）

**Deliverable**：实现一个继承自 `torch.nn.Module` 的 `Embedding` 类。推荐接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```


## 3.5 Pre-Norm Transformer Block

### Problem (`rmsnorm`)：实现 RMSNorm（1 分）

**Deliverable**：将 RMSNorm 实现为一个 `torch.nn.Module`。推荐接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
def forward(self, x: torch.Tensor) -> torch.Tensor
```

测试方式：

1. 实现 `[adapters.run_rmsnorm]`
2. 运行 `uv run pytest -k test_rmsnorm`

### Problem (`positionwise_feedforward`)：实现 Position-Wise FFN（2 分）

**Deliverable**：实现由 SiLU 激活与 GLU 组成的 **SwiGLU** 前馈网络。

要求：

- 令 $d_{\text{ff}} \approx \frac{8}{3} \times d_{\text{model}}$
- 保证 $d_{\text{ff}}$ 是 64 的倍数

测试方式：

1. 实现 `[adapters.run_swiglu]`
2. 运行 `uv run pytest -k test_swiglu`

### Problem (`rope`)：实现 RoPE（2 分）

**Deliverable**：实现一个 `RotaryPositionalEmbedding` 类。推荐接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

测试方式：

1. 实现 `[adapters.run_rope]`
2. 运行 `uv run pytest -k test_rope`

### Problem (`softmax`)：实现 softmax（1 分）

**Deliverable**：编写一个函数，对给定 Tensor 的指定维度执行 softmax。  
测试方式：

1. 实现 `[adapters.run_softmax]`
2. 运行 `uv run pytest -k test_softmax_matches_pytorch`

注意力定义为：

$$
\operatorname{Attention}(Q, K, V)
= \operatorname{softmax}\!\left(\frac{Q^\top K}{\sqrt{d_k}}\right) V
$$

mask 中，`True` 表示允许关注，`False` 表示不允许关注。

### Problem (`scaled_dot_product_attention`)：实现 Scaled Dot-Product Attention（5 分）

**Deliverable**：实现缩放点积注意力函数。

要求：

- key / query 形状：`(batch_size, ..., seq_len, d_k)`
- value 形状：`(batch_size, ..., seq_len, d_v)`
- 支持可选布尔 mask，形状为 `(seq_len, seq_len)`

测试方式：

1. 实现 `[adapters.run_scaled_dot_product_attention]`
2. 运行 `uv run pytest -k test_scaled_dot_product_attention`
3. 再运行 `uv run pytest -k test_4d_scaled_dot_product_attention`

### Problem (`multihead_self_attention`)：实现因果多头自注意力（5 分）

**Deliverable**：将因果多头自注意力实现为一个 `torch.nn.Module`。

要求至少支持：

- `d_model: int`
- `num_heads: int`

并令：

$$
d_k = d_v = d_{\text{model}} / h
$$

测试方式：

1. 实现 `[adapters.run_multihead_self_attention]`
2. 运行 `uv run pytest -k test_multihead_self_attention`


## 3.6 完整的 Transformer LM

### Problem (`transformer_block`)：实现 Transformer block（3 分）

实现第 3.5 节描述、并在图 2 中展示的 pre-norm Transformer block。至少应接受：

- `d_model: int`
- `num_heads: int`
- `d_ff: int`

测试方式：

1. 实现 `[adapters.run_transformer_block]`
2. 运行 `uv run pytest -k test_transformer_block`

### Problem (`transformer_lm`)：实现 Transformer LM（3 分）

按第 3.1 节描述，把 embedding、若干 Transformer blocks 与输出层串接起来。还至少应接受：

- `vocab_size: int`
- `context_length: int`
- `num_layers: int`

测试方式：

1. 实现 `[adapters.run_transformer_lm]`
2. 运行 `uv run pytest -k test_transformer_lm`

### Problem (`transformer_accounting`)：Transformer LM 资源估算（5 分）

矩阵乘法 FLOPs 规则：若 $A \in \mathbb{R}^{m \times n}$、$B \in \mathbb{R}^{n \times p}$，则：

$$
AB \text{ 需要 } 2mnp \text{ FLOPs}
$$

题目要求你分别对 GPT-2 XL 以及其他 GPT-2 规模模型，计算：

- 总参数量
- 仅加载模型需要的内存
- 一次前向传播的矩阵乘法及总 FLOPs
- 各组件 FLOPs 占比
- 上下文长度变化对 FLOPs 的影响

---


# 4 训练 Transformer LM
## 4.1 Cross-Entropy Loss

### Problem (`cross_entropy`)：实现 Cross Entropy

**Deliverable**：编写一个函数，输入预测 logits 与目标标签，计算：
$$
\ell_i = -\log \operatorname{softmax}(o_i)[x_{i+1}]
$$

要求：

- 减去最大值，保证数值稳定
- 尽量在推导中消掉 `log` 与 `exp`
- 支持任意前置 batch-like 维度，并返回 batch 平均损失

测试方式：

1. 实现 `[adapters.run_cross_entropy]`
2. 运行 `uv run pytest -k test_cross_entropy`

若序列长度为 $m$，每个位置损失为 $\ell_1, \dots, \ell_m$，则困惑度为：

$$
\operatorname{perplexity}
= \exp\left(\frac{1}{m}\sum_{i=1}^{m}\ell_i\right)
$$


## 4.2 SGD 优化器

### Problem (`learning_rate_tuning`)：调整学习率（1 分）

把学习率改为 `1e1`、`1e2`、`1e3`，每种只跑 10 次迭代，观察损失变化。  
**Deliverable**：在 lr=10 时，loss 从约 9.25 短暂下降到 8.62，随后迅速飙升到 10^5 到 10^7 量级；在 lr=100 和 lr=1000 时，loss 更快失控，并很快停留在约 9.21 的随机基线附近。这说明过大的学习率会让参数更新幅度过大，模型迅速破坏已有表示，导致训练失败。


## 4.3 AdamW

### Problem (`adamw`)：实现 AdamW（2 分）

**Deliverable**：实现一个继承自 `torch.optim.Optimizer` 的 `AdamW` 类。

测试方式：

1. 实现 `[adapters.get_adamw_cls]`
2. 运行 `uv run pytest -k test_adamw`

### Problem (`adamwAccounting`)：AdamW 资源估算（2 分）

假设全部 Tensor 使用 `float32`。请从以下部分分解峰值内存占用：

- 参数
- 激活
- 梯度
- 优化器状态

并以 `batch_size` 与模型超参数表示。还需估算：

- 单步 AdamW 的 FLOPs
- 在给定 MFU 假设下训练 GPT-2 XL 所需时间


## 4.4 学习率调度

### Problem (`learning_rate_schedule`)：实现带 warmup 的 cosine 调度

实现该调度函数，并完成 `[adapters.get_lr_cosine_schedule]`。测试：

```bash
uv run pytest -k test_get_lr_cosine_schedule
```


## 4.5 梯度裁剪

### Problem (`gradient_clipping`)：实现梯度裁剪（1 分）

编写一个函数，对参数列表中的梯度做原地裁剪。然后实现 `[adapters.run_gradient_clipping]` 并运行：

```bash
uv run pytest -k test_gradient_clipping
```

---


# 5 训练循环
## 5.1 Data Loader

### Problem (`data_loading`)：实现数据加载（2 分）

**Deliverable**：编写一个函数，输入：

- Token ID 的 NumPy 数组
- `batch_size`
- `context_length`
- 目标设备字符串

输出一对 shape 为 `(batch_size, context_length)` 的 Tensor：输入序列与对应 next-token targets。

测试方式：

1. 实现 `[adapters.run_get_batch]`
2. 运行 `uv run pytest -k test_get_batch`

> **低资源/降配提示**
>
> 若数据集太大无法整体加载，可使用 `np.memmap` 或 `np.load(..., mmap_mode='r')` 进行内存映射加载。


## 5.2 Checkpointing

### Problem (`checkpointing`)：实现模型 Checkpoint（1 分）

实现：

```python
def save_checkpoint(model, optimizer, iteration, out)
def load_checkpoint(src, model, optimizer)
```

测试方式：

1. 实现 `[adapters.run_save_checkpoint]`
2. 实现 `[adapters.run_load_checkpoint]`
3. 运行 `uv run pytest -k test_checkpointing`


## 5.3 Training Loop

### Problem (`training_together`)：把所有内容整合起来（4 分）

**Deliverable**：编写一个训练脚本，建议至少支持：

- 配置模型与优化器超参数
- 使用 `np.memmap` 高效加载大型训练集与验证集
- 保存 checkpoint
- 定期记录训练与验证性能

---


# 6 文本生成
## 6.2 常见解码技巧

### Problem (`decoding`)：实现解码（3 分）

**Deliverable**：实现一个 decoder，支持：

- 基于 prompt 生成补全文本
- 控制最大生成 Token 数
- temperature scaling
- top-p / nucleus sampling

---


# 7 实验
## 7.1 实验记录

### Problem (`experiment_log`)：实验记录（3 分）

请为训练与评估代码搭建实验追踪基础设施，以便记录：

- 学习曲线
- 与 steps 对应的 loss
- 与 wallclock time 对应的 loss
- 实验配置与尝试历史


## 7.2 TinyStories

### Problem (`learning_rate`)：调学习率（3 分）

**Deliverable**：

- 多组学习率下的学习曲线
- 超参数搜索策略说明
- 一个在 TinyStories 上验证损失不高于 **1.45** 的模型

### Problem (`batch_size_experiment`)：batch size 变化实验（1 分）

把 batch size 从 1 一直试到显存上限。  
**Deliverable**：不同 batch size 下的学习曲线，并讨论其对训练的影响。

### Problem (`generate`)：生成文本（1 分）

**Deliverable**：至少 **256 个 Token** 的生成文本（或直到第一个 `<|endoftext|>`），并评论其流畅度及影响因素。


## 7.3 消融实验与架构修改

### Problem (`layer_norm_ablation`)：移除 RMSNorm 并训练（1 分）

**Deliverable**：移除 RMSNorm 前后的学习曲线对比，并简述结论。

Pre-norm 形式：

$$
z = x + \operatorname{MultiHeadedSelfAttention}(\operatorname{RMSNorm}(x))
$$

$$
y = z + \operatorname{FFN}(\operatorname{RMSNorm}(z))
$$

Post-norm 形式：

$$
z = \operatorname{RMSNorm}(x + \operatorname{MultiHeadedSelfAttention}(x))
$$

$$
y = \operatorname{RMSNorm}(z + \operatorname{FFN}(z))
$$

### Problem (`pre_norm_ablation`)：实现 post-norm 并训练（1 分）

**Deliverable**：post-norm 与 pre-norm 的学习曲线对比。

### Problem (`no_pos_emb`)：实现 NoPE（1 分）

**Deliverable**：RoPE 与 NoPE 的学习曲线对比。

无门控 SiLU 前馈网络：

$$
\operatorname{FFN}_{\text{SiLU}}(x) = W_2 \operatorname{SiLU}(W_1 x)
$$

### Problem (`swiglu_ablation`)：SwiGLU vs SiLU（1 分）

**Deliverable**：SwiGLU 与 SiLU 前馈网络的学习曲线对比，并简述结论。


## 7.4 在 OpenWebText 上运行

### Problem (`main_experiment`)：在 OWT 上做实验（2 分）

**Deliverable**：

- OpenWebText 上的学习曲线
- 与 TinyStories 损失的对比分析
- OpenWebText LM 的生成文本与流畅度讨论


## 7.5 你自己的改动 + 排行榜

### Problem (`leaderboard`)：排行榜（6 分）

**Deliverable**：报告最终验证损失、学习曲线（横轴应明确展示墙钟时间且小于 1.5 小时），并说明你做了哪些改动。

---

