# CS336 作业 1（基础篇）：构建 Transformer 语言模型

**版本**：1.0.6  
**CS336 教学团队**  
**2025 年春季**

---

# 1 作业概览

在本次作业中，你将**从零开始**构建训练标准 Transformer 语言模型（LM）所需的全部组件，并训练若干模型。

## 1.1 你将实现的内容

1. 字节对编码（Byte-Pair Encoding, **BPE**）Tokenizer（见第 2 节）
2. Transformer 语言模型（Language Model, **LM**）（见第 3 节）
3. 交叉熵损失（Cross-Entropy Loss）函数与 AdamW 优化器（见第 4 节）
4. 训练循环（training loop），并支持模型状态与优化器状态的序列化和加载（见第 5 节）

## 1.2 你将运行的内容

1. 在 TinyStories 数据集上训练一个 BPE Tokenizer。
2. 使用你训练好的 Tokenizer 对数据集进行编码，将文本转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练一个 Transformer LM。
4. 使用训练好的 Transformer LM 生成样本并评估困惑度（perplexity）。
5. 在 OpenWebText 上训练模型，并将你达到的 perplexity 提交到排行榜。

## 1.3 允许使用的内容

我们希望你**从头实现**这些组件。特别地，你**不得**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的定义，以下内容除外：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）
- `torch.optim.Optimizer` 基类

除此之外，你可以使用其他任意 PyTorch 定义。如果你不确定某个函数或类是否允许使用，可以在 Slack 上提问。判断标准通常是：它是否违背了本作业“从零实现”的精神。

注 1：完整容器类列表见 PyTorch 官方文档。

## 1.4 关于 AI 工具的声明

允许使用 ChatGPT 等大语言模型（LLM）来询问**底层编程问题**或**关于语言模型的高层概念问题**，但**禁止**直接使用它们来完成题目本身。

我们强烈建议你在完成作业时关闭 IDE 中的 AI 自动补全功能（例如 Cursor Tab、GitHub Copilot）。非 AI 的自动补全（例如函数名补全）当然可以使用。教学团队发现，AI 自动补全会显著削弱你对作业内容的深入理解。

## 1.5 代码仓库结构

作业代码与本文档均可在 GitHub 获取：

`github.com/stanford-cs336/assignment1-basics`

请先 `git clone` 仓库。如有更新，课程组会通知你执行 `git pull` 获取最新内容。

1. `cs336_basics/*`：你编写代码的目录。这里默认**没有现成实现**，你可以完全从零开始。
2. `adapters.py`：这里定义了你的代码必须提供的功能接口。对于每项功能（例如 scaled dot-product attention），你只需在对应适配器函数中调用你自己的实现（例如 `run_scaled_dot_product_attention`）。注意：`adapters.py` 中不应包含实质性逻辑，它只是胶水代码。
3. `test_*.py`：这里包含你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），这些测试会调用 `adapters.py` 中定义的 hook。**不要修改测试文件。**

## 1.6 提交方式

你需要向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面题，请使用排版良好的文档。
- `code.zip`：包含你编写的全部代码。

如果要提交到排行榜，请向以下仓库提交 PR：

`github.com/stanford-cs336/assignment1-basics-leaderboard`

详细说明见排行榜仓库中的 `README.md`。

## 1.7 数据集获取

本作业将使用两个已预处理数据集：TinyStories [Eldan and Li, 2023] 和 OpenWebText [Gokaslan et al., 2019]。它们都是单个大型纯文本文件。

- 如果你是在课程机器上完成作业，可在任意非 head node 机器的 `/data` 目录找到这些文件。
- 如果你在本地完成，可使用 `README.md` 中给出的命令下载。

> **低资源/降配提示：总说明**
>
> 在整门课的作业讲义中，我们都会提供一些建议，帮助你在 GPU 资源较少甚至没有 GPU 的情况下完成作业。例如，我们有时会建议缩小数据集规模或模型规模，或者解释如何在 MacOS 集成 GPU 或 CPU 上运行训练代码。  
> 这些“低资源提示”会放在蓝色提示框中。即便你是 Stanford 在读学生、可以访问课程机器，这些建议通常也能帮助你更快迭代、节省时间，因此推荐认真阅读。

> **低资源/降配提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用助教参考实现，我们可以在配备 36 GB RAM 的 Apple M3 Max 上，在 Metal GPU（MPS）下 5 分钟内训练出一个能生成相当流畅文本的语言模型；若使用 CPU，大约需要 30 分钟。  
> 如果这些术语你不熟悉也没关系。你只需要知道：如果你的笔记本足够新，并且实现正确且高效，你仍然可以训练出一个能够生成简单儿童故事、流畅度尚可的小型 LM。  
> 后文会说明在 CPU 或 MPS 上需要做哪些调整。

---

# 2 字节对编码（BPE）Tokenizer

在本作业的第一部分中，我们将训练并实现一个**字节级**字节对编码（Byte-Pair Encoding, **BPE**）Tokenizer [Sennrich et al., 2016; Wang et al., 2019]。具体而言，我们会将任意 Unicode 字符串表示为字节序列，并在该字节序列上训练 BPE Tokenizer。随后，我们将使用该 Tokenizer 把文本（字符串）编码为 Token（整数序列），用于语言模型训练。

## 2.1 Unicode 标准

Unicode 是一种文本编码标准，它将字符映射为整数**码点（code point）**。截至 Unicode 16.0（2024 年 9 月发布），该标准在 168 个书写系统中定义了 154,998 个字符。

例如：

- 字符 `"s"` 的码点是 115，通常记作 `U+0073`
- 字符 `"୤"` 的码点是 29275

在 Python 中，你可以使用 `ord()` 将单个 Unicode 字符转换为整数表示；使用 `chr()` 将整数码点转换为对应字符。

```python
>>> ord('୤')
29275
>>> chr(29275)
'୤'
```

### Problem (`unicode1`)：理解 Unicode（1 分）

(a) `chr(0)` 返回的 Unicode 字符是什么？  
**Deliverable**：一句话回答。

(b) 该字符的字符串表示（`__repr__()`）与其打印表示有何不同？  
**Deliverable**：一句话回答。

(c) 当该字符出现在文本中时会发生什么？你可以在 Python 解释器中尝试以下代码，并观察其行为是否符合预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**Deliverable**：一句话回答。

## 2.2 Unicode 编码

虽然 Unicode 标准定义了字符到码点（整数）的映射，但如果直接在 Unicode 码点上训练 Tokenizer，会非常不现实，因为词表会过大（约 15 万项），而且非常稀疏（许多字符极少出现）。

因此，我们改用 **Unicode Encoding**，即把一个 Unicode 字符编码成一个**字节序列**。Unicode 标准本身定义了三种常见编码：

- UTF-8
- UTF-16
- UTF-32

其中 UTF-8 是互联网的主流编码（超过 98% 的网页采用 UTF-8）。

在 Python 中：

- 用 `encode()` 将 Unicode 字符串编码为 UTF-8 字节串
- 对 `bytes` 对象迭代或调用 `list()` 可查看底层字节值
- 用 `decode()` 将 UTF-8 字节串解码为 Unicode 字符串

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字节串的字节值（0 到 255 的整数）
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节并不一定对应一个 Unicode 字符
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

把 Unicode 码点转换为字节序列（例如使用 UTF-8）之后，本质上是把一个取值范围在 0 到 154,997 的整数序列，转换成一个取值范围在 0 到 255 的字节值序列。长度为 256 的字节词表显然更容易处理。

采用字节级 Tokenization 时，我们不需要担心 OOV（out-of-vocabulary）问题，因为任何输入文本都可以表示为 0 到 255 的整数序列。

### Problem (`unicode2`)：Unicode 编码（3 分）

(a) 为什么训练 Tokenizer 时更倾向于使用 UTF-8 编码字节，而不是 UTF-16 或 UTF-32？对若干输入字符串比较这些编码的输出，可能会有帮助。  
**Deliverable**：1 至 2 句话回答。

(b) 下面这个函数本意是把 UTF-8 字节串解码为 Unicode 字符串，但它是**错误的**。为什么？请给出一个会产生错误输出的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**Deliverable**：给出一个输入字节串示例，并用一句话说明为什么该函数是错误的。

(c) 给出一个**无法解码为任何 Unicode 字符**的两字节序列。  
**Deliverable**：给出示例并用一句话解释。

## 2.3 Subword Tokenization

虽然字节级 Tokenization 能缓解词级 Tokenizer 面临的 OOV 问题，但按字节切分会使输入序列变得非常长，从而拖慢模型训练。比如，一个包含 10 个单词的句子，在词级 LM 中可能只对应 10 个 Token，而在字符级模型中可能对应 50 个甚至更多 Token。更长的序列意味着每一步都需要更多计算，也会使长期依赖问题更加严重。

**Subword Tokenization** 介于词级和字节级之间。字节级 Tokenizer 的词表大小固定为 256，而 Subword Tokenizer 用更大的词表换取更高的压缩率。比如，如果字节序列 `b'the'` 在训练语料中经常出现，那么把它加入词表就可以把原本 3 个 Token 压缩成 1 个 Token。

那么，哪些 Subword 单元应该加入词表？Sennrich 等人 [2016] 提出使用 **BPE**（源于 Gage, 1994 的压缩算法）：它迭代地把最常见的一对字节替换（merge）成一个新的、尚未使用的索引。这样，词表中的 Subword Token 会自然偏向最大化对输入字节序列的压缩；如果某个词在语料中出现足够频繁，它最终可能整体变成一个单独的 Subword 单元。

使用 BPE 构建词表的 Subword Tokenizer 通常称为 **BPE Tokenizer**。本作业中，我们实现的是**字节级 BPE Tokenizer**：词表项既可以是单字节，也可以是若干字节合并后的序列，从而同时兼顾 OOV 处理能力与较短的输入序列长度。构建 BPE 词表的过程通常也称为 BPE Tokenizer 的“训练”。

## 2.4 BPE Tokenizer 训练

BPE Tokenizer 的训练过程包含三大步骤。

### 2.4.1 词表初始化

Tokenizer 词表是从**字节串 Token** 到**整数 ID** 的一一映射。由于我们训练的是字节级 BPE Tokenizer，初始词表就是全部可能字节的集合。因为一共有 256 种字节取值，所以初始词表大小为 256。

### 2.4.2 预分词

在有了词表之后，理论上你可以直接统计文本中相邻字节对的出现频率，并从最高频字节对开始合并。但这样代价很高，因为每次 merge 都要完整扫描整个语料。

此外，直接在整份语料上跨边界合并字节，可能会得到仅在标点上不同的 Token，例如 `dog!` 与 `dog.`。它们会得到完全不同的 Token ID，尽管语义非常相近。

为避免这一点，我们会先对语料做**预分词（pre-tokenization）**。你可以把它理解为一种较粗粒度的分词，用于帮助我们统计字符对的共现频率。

Sennrich 等人 [2016] 的原始实现只是简单按空白切分，即 `s.split(" ")`。而本作业中，我们改用 GPT-2 风格的正则预分词器 [Radford et al., 2019]：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

```python
>>> import regex as re  # 需要安装 regex 包
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

实际代码中应使用 `re.finditer`，避免一次性把所有 pre-token 全部放入内存。

### 2.4.3 计算 BPE merges

当我们把输入文本转换成 pre-token，并将每个 pre-token 表示为 UTF-8 字节序列之后，就可以开始计算 BPE merges。

高层思路如下：

1. 统计所有相邻字节对的频率。
2. 找出频率最高的一对字节 `("A", "B")`。
3. 将其每次出现都合并为新 Token `"AB"`。
4. 把新 Token 加入词表。
5. 重复上述过程。

若多个字节对频率并列最高，则要求**确定性**地打破平局：优先选择**字典序更大**的字节对。例如：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

### 2.4.4 Special Tokens

有些字符串（例如 `<|endoftext|>`）常用于编码元数据，例如文档边界。编码文本时，通常希望把这些字符串视为**Special Tokens**，也就是说，它们不应再被拆分成多个 Token，而应始终作为一个整体保留。

### Example (`bpe_example`)：BPE 训练示例

考虑如下语料：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并假设词表中还包含一个 Special Token：`<|endoftext|>`。

若只取前 6 次 merge，则最终一些新增词表元素会是：

```text
[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]
```

于是单词 `newest` 会被切分为：

```text
[ne, west]
```

## 2.5 BPE Tokenizer 训练实验

现在，让我们在 TinyStories 数据集上训练一个**字节级 BPE Tokenizer**。

### 2.5.1 并行化预分词

预分词往往是训练的主要瓶颈。可以使用 `multiprocessing` 对其并行化，并确保 chunk 边界落在 Special Token 的起始位置。课程组提供了可直接复用的示例代码：

`https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py`

### 2.5.2 在预分词前移除 Special Tokens

在使用正则进行预分词之前，应先按 Special Tokens 切分语料，确保 merge 不会跨越文档边界。测试 `test_train_bpe_special_tokens` 会检查这一点。

### 2.5.3 优化 merge 步骤

若为所有 pair count 建立索引并采用**增量更新**，而不是每轮都重新遍历全部 pair，BPE 训练可以显著提速。不过 merge 本身在 Python 中**不易并行化**。

> **低资源/降配提示：性能分析**
>
> 推荐使用 `cProfile` 或 `scalene` 找出实现瓶颈。

> **低资源/降配提示：先“降配”调试**
>
> 建议先在 TinyStories 验证集等更小的“调试数据集”上开发与调试，再迁移到完整数据。

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
**Deliverable**：1 至 2 句话回答训练耗时、内存占用、最长 Token 及其合理性。

(b) 对代码做 profiling，指出最耗时部分。  
**Deliverable**：1 至 2 句话回答。

### Problem (`train_bpe_expts_owt`)：在 OpenWebText 上训练 BPE（2 分）

(a) 在 OpenWebText 上训练一个字节级 BPE Tokenizer，最大词表大小设为 **32,000**。  
**资源要求**：不使用 GPU 时 ≤ 12 小时，内存 ≤ 100 GB。  
**Deliverable**：1 至 2 句话回答最长 Token 及其合理性。

(b) 比较在 TinyStories 与 OpenWebText 上训练出的 Tokenizer。  
**Deliverable**：1 至 2 句话回答。

## 2.6 BPE Tokenizer：编码与解码

### 2.6.1 文本编码

BPE 编码文本的过程与训练 BPE 词表的过程高度对应：

1. 预分词
2. 将每个 pre-token 表示为 UTF-8 字节序列
3. 按创建顺序依次应用 merges

### Example (`bpe_encoding`)：BPE 编码示例

假设输入字符串为 `'the cat ate'`，则编码流程是先预分词为 `['the', ' cat', ' ate']`，再依次应用训练中学到的 merges，最终得到整数序列 `[9, 7, 1, 5, 10, 3]`。

**Special Tokens**：编码时还应正确处理用户定义的 Special Tokens。  
**内存考虑**：为支持超大文件，应提供流式 / chunked 编码能力，并确保 Token 不会跨 chunk 边界。

### 2.6.2 文本解码

将 Token ID 序列解码回文本时，只需：

1. 查词表得到对应字节序列
2. 拼接字节
3. 用 UTF-8 解码

若结果不是合法 Unicode，应将非法字节替换为 `U+FFFD`。Python 中可使用 `errors='replace'`。

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
**Deliverable**：1 至 2 句话回答。

(c) 估算 Tokenizer 吞吐量，并推算编码 The Pile（825 GB）所需时间。  
**Deliverable**：1 至 2 句话回答。

(d) 将训练集和验证集编码为整数 Token ID 序列，建议保存为 `uint16` NumPy 数组。为什么 `uint16` 合适？  
**Deliverable**：1 至 2 句话回答。

---

# 3 Transformer 语言模型架构

![图 1：Transformer 语言模型总览](./images/figure1_transformer_lm_overview.png)

*图 1：我们的 Transformer 语言模型总体结构。*

![图 2：Pre-norm Transformer block](./images/figure2_pre_norm_transformer_block.png)

*图 2：Pre-norm Transformer block。*

语言模型接收一个**批量化的整数 Token ID 序列**作为输入，即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`，并输出一个关于词表的**批量化归一化概率分布**，即形状为 `(batch_size, sequence_length, vocab_size)` 的 Tensor。该分布表示：对于输入序列中每个位置，模型对“下一个词”的预测概率。

训练时，我们用这些 next-word predictions 与真实下一个词之间的差异来计算 Cross-Entropy Loss。推理生成时，我们取最后一个时间步的预测分布，生成下一个 Token，再把新 Token 追加到输入序列中，不断循环。

## 3.1 Transformer LM

给定 Token ID 序列，Transformer 语言模型首先通过**输入嵌入层（input embedding）**将 Token ID 映射为稠密向量，随后把这些嵌入送入 `num_layers` 个 Transformer blocks，最后通过一个可学习的线性投影（也称**输出嵌入**或 **LM head**）得到预测下一个 Token 的 logits。结构示意见图 1。

### 3.1.1 Token Embeddings

Token embedding 层接收形状为 `(batch_size, sequence_length)` 的整数 Tensor，并输出形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

### 3.1.2 Pre-norm Transformer Block

标准 decoder-only Transformer LM 由 `num_layers` 个相同层组成。每个 block 接收形状 `(batch_size, sequence_length, d_model)` 的输入，并输出相同形状的 Tensor。它通过**自注意力机制（Self-Attention）**聚合序列信息，并通过**前馈网络（Feed-Forward Network, FFN）**做非线性变换。

## 3.2 输出归一化与输出嵌入

经过 `num_layers` 个 Transformer blocks 后，我们还需要把最终激活转换为词表上的分布。

本作业实现的是 **pre-norm Transformer block**，因此在最后一个 Transformer block 之后，还要再做一次 Layer Normalization，以确保输出尺度合适。随后使用一个标准可学习线性变换，把输出映射为下一个 Token 的 logits。

## 3.3 说明：Batching、`einsum` 与高效计算

在整个 Transformer 中，我们会反复对很多“类 batch 维度”执行同样的运算，例如：

- batch 中的不同样本
- 序列中的不同位置
- 多头注意力中的不同 heads

为了充分利用 GPU 并保持代码可读性，通常需要一种既高效又表达清晰的张量操作方式。课程强烈建议学习并使用 `einsum` 记号。

### Example (`einstein_example1`)：用 `einops.einsum` 做批量矩阵乘法

```python
import torch
from einops import rearrange, einsum

Y = D @ A.T
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

### Example (`einstein_example2`)：用 `einops.rearrange` 做广播运算

```python
images = torch.randn(64, 128, 128, 3)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

### Example (`einstein_example3`)：像素混合

```python
channels_last = torch.randn(64, 32, 32, 3)
B = torch.randn(32 * 32, 32 * 32)

channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=32, width=32
)
```

### 3.3.1 数学记号与内存布局

许多机器学习论文使用**行向量**记法，对应的线性变换是：

$$
y = xW^\top
$$

在线性代数中更常见的是**列向量**记法：

$$
y = Wx
$$

本作业的数学推导统一采用**列向量**记法，但你在代码中仍需注意 PyTorch 的 row-major 内存布局。若使用 `einsum`，这个问题通常会简单得多。

## 3.4 基础组件：Linear 与 Embedding 模块

### 3.4.1 参数初始化

本作业中，请直接使用如下初始化：

- **Linear 权重**：$\mathcal{N}\!\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)$，并在 $[-3\sigma, 3\sigma]$ 上截断
- **Embedding 权重**：$\mathcal{N}(0, 1)$，并在 $[-3, 3]$ 上截断
- **RMSNorm 增益参数**：初始化为 1

请使用 `torch.nn.init.trunc_normal_`。

### 3.4.2 Linear 模块

Linear 层执行的线性变换为：

$$
y = Wx
$$

注意：本作业中的 Linear **不包含 bias**。

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

### 3.4.3 Embedding 模块

Embedding 层将整数 Token ID 映射到 $d_{\text{model}}$ 维向量空间。

### Problem (`embedding`)：实现 Embedding 模块（1 分）

**Deliverable**：实现一个继承自 `torch.nn.Module` 的 `Embedding` 类。推荐接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

## 3.5 Pre-Norm Transformer Block

每个 Transformer block 含两个子层：

- 多头自注意力（Multi-Head Self-Attention）
- 位置前馈网络（Position-Wise Feed-Forward）

原始 Transformer 使用 post-norm；本作业采用更稳定的 **pre-norm** 形式。

### 3.5.1 RMSNorm

给定激活向量 $a \in \mathbb{R}^{d_{\text{model}}}$，RMSNorm 定义为：

$$
\operatorname{RMSNorm}(a_i) = \frac{a_i}{\operatorname{RMS}(a)} g_i
$$

其中：

$$
\operatorname{RMS}(a) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} a_i^2 + \varepsilon}
$$

实现时应先把输入 upcast 到 `torch.float32`，再在输出时 cast 回原始 dtype。

### Problem (`rmsnorm`)：实现 RMSNorm（1 分）

**Deliverable**：将 RMSNorm 实现为一个 `torch.nn.Module`。推荐接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
def forward(self, x: torch.Tensor) -> torch.Tensor
```

测试方式：

1. 实现 `[adapters.run_rmsnorm]`
2. 运行 `uv run pytest -k test_rmsnorm`

### 3.5.2 Position-Wise Feed-Forward Network

![图 3：SiLU 与 ReLU 激活函数对比](./images/figure3_silu_vs_relu.png)

*图 3：SiLU（又称 Swish）与 ReLU 激活函数的比较。*

现代语言模型通常使用 **SwiGLU**。其中 SiLU 定义为：

$$
\operatorname{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

GLU 定义为：

$$
\operatorname{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x
$$

合并后得到：

$$
\operatorname{FFN}(x)
= W_2 \big(\operatorname{SiLU}(W_1 x) \odot W_3 x \big)
$$

其中通常有：

$$
d_{\text{ff}} = \frac{8}{3} d_{\text{model}}
$$

### Problem (`positionwise_feedforward`)：实现 Position-Wise FFN（2 分）

**Deliverable**：实现由 SiLU 激活与 GLU 组成的 **SwiGLU** 前馈网络。

要求：

- 令 $d_{\text{ff}} \approx \frac{8}{3} \times d_{\text{model}}$
- 保证 $d_{\text{ff}}$ 是 64 的倍数

测试方式：

1. 实现 `[adapters.run_swiglu]`
2. 运行 `uv run pytest -k test_swiglu`

### 3.5.3 相对位置嵌入：RoPE

我们使用 **旋转位置嵌入（Rotary Position Embeddings, RoPE）** [Su et al., 2021]。

对位于位置 $i$ 的 query：

$$
q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d
$$

施加旋转矩阵 $R_i$ 后：

$$
q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}
$$

第 $k$ 对维度的旋转角为：

$$
\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}
$$

对应的 $2 \times 2$ 旋转块为：

$$
R_i^k =
\begin{bmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{bmatrix}
$$

RoPE 只作用于 query 和 key，不作用于 value。

### Problem (`rope`)：实现 RoPE（2 分）

**Deliverable**：实现一个 `RotaryPositionalEmbedding` 类。推荐接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

测试方式：

1. 实现 `[adapters.run_rope]`
2. 运行 `uv run pytest -k test_rope`

### 3.5.4 Scaled Dot-Product Attention

softmax 定义为：

$$
\operatorname{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}
$$

实现时应使用减去最大值的技巧确保数值稳定。

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

### 3.5.5 因果多头自注意力

多头注意力定义为：

$$
\operatorname{MultiHead}(Q, K, V)
= \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)
$$

其中：

$$
\text{head}_i = \operatorname{Attention}(Q_i, K_i, V_i)
$$

自注意力可写为：

$$
\operatorname{MultiHeadSelfAttention}(x)
= W_O \operatorname{MultiHead}(W_Q x, W_K x, W_V x)
$$

实现时必须使用**因果掩码（causal mask）**，阻止模型看到未来位置。

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

Transformer block 的第一个子层应实现：

$$
y = x + \operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x))
$$

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

现在，我们已经具备了：

- 数据预处理能力（Tokenizer）
- 模型本身（Transformer）

还差的是训练基础设施：

- **Loss**：定义损失函数
- **Optimizer**：定义优化器
- **Training loop**：完成数据加载、checkpoint 保存与训练调度

## 4.1 Cross-Entropy Loss

给定训练集 $D$，标准交叉熵损失定义为：

$$
\ell(\theta; D)
= \frac{1}{|D|m}
\sum_{x \in D}
\sum_{i=1}^{m}
-\log p_\theta(x_{i+1} \mid x_{1:i})
$$

若位置 $i$ 的 logits 为 $o_i \in \mathbb{R}^{\text{vocab\_size}}$，则：

$$
p(x_{i+1} \mid x_{1:i})
= \operatorname{softmax}(o_i)[x_{i+1}]
= \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])}
$$

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

随机梯度下降（SGD）更新为：

$$
\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t)
$$

其中 $B_t$ 是 batch，$\alpha_t$ 是学习率。

### 4.2.1 在 PyTorch 中实现 SGD

示例中的衰减式 SGD 为：

$$
\theta_{t+1}
= \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t)
$$

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
```

### Problem (`learning_rate_tuning`)：调整学习率（1 分）

把学习率改为 `1e1`、`1e2`、`1e3`，每种只跑 10 次迭代，观察损失变化。  
**Deliverable**：1 至 2 句话说明你的观察。

## 4.3 AdamW

现代语言模型通常使用 **AdamW** [Loshchilov and Hutter, 2019]。它在 Adam 的基础上把 **weight decay** 从梯度更新中解耦出来，并为每个参数维护一阶矩与二阶矩估计。

### Algorithm 1：AdamW Optimizer

1. 初始化参数 $\theta$
2. 令 $m \leftarrow 0$
3. 令 $v \leftarrow 0$
4. 对每个时间步 $t=1,\dots,T$：
5. 采样 batch $B_t$
6. 计算梯度 $g \leftarrow \nabla_\theta \ell(\theta; B_t)$
7. 更新一阶矩：

$$
m \leftarrow \beta_1 m + (1 - \beta_1) g
$$

8. 更新二阶矩：

$$
v \leftarrow \beta_2 v + (1 - \beta_2) g^2
$$

9. 偏置修正后的学习率：

$$
\alpha_t = \alpha \frac{\sqrt{1 - (\beta_2)^t}}{1 - (\beta_1)^t}
$$

10. 更新参数：

$$
\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v} + \epsilon}
$$

11. 应用权重衰减：

$$
\theta \leftarrow \theta - \alpha \lambda \theta
$$

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

本作业采用 cosine annealing 调度。

Warm-up 阶段：

$$
\alpha_t = \frac{t}{T_w} \alpha_{\max}
$$

Cosine annealing 阶段：

$$
\alpha_t
= \alpha_{\min}
+ \frac{1}{2}
\left(
1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)
\right)
(\alpha_{\max} - \alpha_{\min})
$$

Post-annealing 阶段：

$$
\alpha_t = \alpha_{\min}
$$

### Problem (`learning_rate_schedule`)：实现带 warmup 的 cosine 调度

实现该调度函数，并完成 `[adapters.get_lr_cosine_schedule]`。测试：

```bash
uv run pytest -k test_get_lr_cosine_schedule
```

## 4.5 梯度裁剪

若梯度总体 $\ell_2$ 范数 $\|g\|_2$ 大于阈值 $M$，则把梯度缩放为：

$$
\frac{M}{\|g\|_2 + \epsilon}
$$

其中 $\epsilon = 10^{-6}$。

### Problem (`gradient_clipping`)：实现梯度裁剪（1 分）

编写一个函数，对参数列表中的梯度做原地裁剪。然后实现 `[adapters.run_gradient_clipping]` 并运行：

```bash
uv run pytest -k test_gradient_clipping
```

---

# 5 训练循环

## 5.1 Data Loader

编码后的数据可视作一个长序列：

$$
x = (x_1, \dots, x_n)
$$

数据加载器把它切成训练批次。对 $B=1, m=3$ 的一个例子：

```text
([x2, x3, x4], [x3, x4, x5])
```

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

checkpoint 至少应保存：

- 模型参数
- 优化器状态
- 当前迭代编号

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

## 6.1 解码

单步解码可表示为：

$$
P(x_{t+1} = i \mid x_{1:t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}
$$

其中：

$$
v = \operatorname{TransformerLM}(x_{1:t})_t \in \mathbb{R}^{\text{vocab\_size}}
$$

不断重复该过程，直到生成 `<|endoftext|>` 或达到最大长度。

## 6.2 常见解码技巧

温度缩放：

$$
\operatorname{softmax}(v, \tau)_i
= \frac{\exp(v_i / \tau)}{\sum_{j=1}^{|\text{vocab}|} \exp(v_j / \tau)}
$$

Top-p / nucleus sampling：

$$
P(x_{t+1} = i \mid q) =
\begin{cases}
\dfrac{q_i}{\sum_{j \in V(p)} q_j}, & i \in V(p) \\
0, & \text{otherwise}
\end{cases}
$$

其中 $V(p)$ 是累计概率刚达到阈值 $p$ 的最小词表子集。

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

### Example (`tinystories_example`)：TinyStories 样例

从前有一个小男孩叫 Ben。Ben 很喜欢探索周围的世界。他看到许多令人惊叹的东西，比如商店里陈列着的漂亮花瓶。一天，Ben 走过商店时，发现了一个非常特别的花瓶。Ben 一看到它就惊呆了。

推荐起始超参数：

- `vocab_size = 10000`
- `context_length = 256`
- `d_model = 512`
- `d_ff = 1344`
- `theta = 10000`
- 4 层、16 头
- 总处理 Token 数约为 327,680,000

### Problem (`learning_rate`)：调学习率（3 分）

**Deliverable**：

- 多组学习率下的学习曲线
- 超参数搜索策略说明
- 一个在 TinyStories 上验证损失不高于 **1.45** 的模型

### Problem (`batch_size_experiment`)：batch size 变化实验（1 分）

把 batch size 从 1 一直试到显存上限。  
**Deliverable**：不同 batch size 下的学习曲线，并讨论其对训练的影响。

### Example (`ts_generate_example`)：TinyStories 生成样例

从前，有一个漂亮的小女孩叫 Lily。她很喜欢吃口香糖。一天，Lily 的妈妈让她帮忙做晚饭。Lily 非常兴奋……

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

### Example (`owt_example`)：OpenWebText 样例

OpenWebText 的文本通常比 TinyStories 更真实、更复杂、更多样，覆盖网页爬取语料中的各类主题与文体。

### Problem (`main_experiment`)：在 OWT 上做实验（2 分）

**Deliverable**：

- OpenWebText 上的学习曲线
- 与 TinyStories 损失的对比分析
- OpenWebText LM 的生成文本与流畅度讨论

## 7.5 你自己的改动 + 排行榜

排行榜规则：

- 训练必须在 **1.5 小时 H100** 内完成
- 只能使用课程提供的 OpenWebText 训练集

### Problem (`leaderboard`)：排行榜（6 分）

**Deliverable**：报告最终验证损失、学习曲线（横轴应明确展示墙钟时间且小于 1.5 小时），并说明你做了哪些改动。

---

# 参考文献

为避免检索歧义，以下书目信息保留原始英文格式：

- Ronen Eldan and Yuanzhi Li. *TinyStories: How small can language models be and still speak coherent English?* 2023. arXiv:2305.07759.
- Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. *OpenWebText corpus.* 2019.
- Rico Sennrich, Barry Haddow, and Alexandra Birch. *Neural machine translation of rare words with subword units.* ACL, 2016.
- Changhan Wang, Kyunghyun Cho, and Jiatao Gu. *Neural machine translation with byte-level subwords.* 2019.
- Philip Gage. *A new algorithm for data compression.* C Users Journal, 1994.
- Alec Radford et al. *Language models are unsupervised multitask learners.* 2019.
- Alec Radford et al. *Improving language understanding by generative pre-training.* 2018.
- Ashish Vaswani et al. *Attention is all you need.* NeurIPS, 2017.
- Toan Q. Nguyen and Julian Salazar. *Transformers without tears: Improving the normalization of self-attention.* 2019.
- Ruibin Xiong et al. *On layer normalization in the Transformer architecture.* ICML, 2020.
- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. *Layer normalization.* 2016.
- Hugo Touvron et al. *Llama: Open and efficient foundation language models.* 2023.
- Biao Zhang and Rico Sennrich. *Root mean square layer normalization.* NeurIPS, 2019.
- Aaron Grattafiori et al. *The llama 3 herd of models.* 2024.
- An Yang et al. *Qwen2.5 technical report.* 2024.
- Aakanksha Chowdhery et al. *PaLM: Scaling language modeling with pathways.* 2022.
- Dan Hendrycks and Kevin Gimpel. *Bridging nonlinearities and stochastic regularizers with gaussian error linear units.* 2016.
- Stefan Elfwing, Eiji Uchibe, and Kenji Doya. *Sigmoid-weighted linear units for neural network function approximation in reinforcement learning.* 2017.
- Yann N. Dauphin et al. *Language modeling with gated convolutional networks.* 2017.
- Noam Shazeer. *GLU variants improve transformer.* 2020.
- Jianlin Su et al. *RoFormer: Enhanced transformer with rotary position embedding.* 2021.
- Diederik P. Kingma and Jimmy Ba. *Adam: A method for stochastic optimization.* ICLR, 2015.
- Ilya Loshchilov and Frank Hutter. *Decoupled weight decay regularization.* ICLR, 2019.
- Tom B. Brown et al. *Language models are few-shot learners.* NeurIPS, 2020.
- Jared Kaplan et al. *Scaling laws for neural language models.* 2020.
- Jordan Hoffmann et al. *Training compute-optimal large language models.* 2022.
- Ari Holtzman et al. *The curious case of neural text degeneration.* ICLR, 2020.
- Yao-Hung Hubert Tsai et al. *Transformer dissection: An unified understanding for transformer’s attention via the lens of kernel.* EMNLP-IJCNLP, 2019.
