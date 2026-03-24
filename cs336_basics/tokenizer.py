import regex as re
from collections import Counter

# 这是 GPT-2 使用的经典正则表达式（来自 OpenAI 的 tiktoken [cite: 154]）
# 它的作用是聪明地把文本切碎，比如把 "i'll" 切成 "i" 和 "'ll"，把连续的空格切到一起。
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_word(word_tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    在一个 token 序列里，把指定的 pair 从左到右进行贪心合并。
    """
    left, right = pair
    merged = []
    i = 0
    
    while i < len(word_tokens):
        if i < len(word_tokens) - 1 and word_tokens[i] == left and word_tokens[i + 1] == right:
            # 在匹配到目标pair同时防止越界
            merged.append(left + right)
            i += 2  
            # 匹配到之后拼接，同时索引 + 2 
        else:
            merged.append(word_tokens[i])
            i += 1
            # 没匹配到，则保存单个byte，往后继续
    return tuple(merged)

from collections import Counter

def get_pair_counts(word_freq: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    遍历全局词频字典，统计所有相邻 token pair 的总频次。
    """
    pair_counts = Counter()
    
    for token_seq, freq in word_freq.items():
        # 遍历当前单词的所有相邻对
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            # 累加这个单词在全局出现的频次
            pair_counts[pair] += freq
            
    return pair_counts


def apply_merge(
    word_freq: dict[tuple[bytes, ...], int], 
    pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    """
    遍历全局词频字典，把指定的 pair 全部合并，并返回一个全新的词频字典。
    """
    new_word_freq = {}
    
    for token_seq, freq in word_freq.items():
        # 生成合并后的新序列
        new_seq = merge_word(token_seq, pair)
        
        # 把新序列放进新字典，并累加频次
        # .get(new_seq, 0) 的意思是：如果字典里还没有这个 key，就默认返回 0
        new_word_freq[new_seq] = new_word_freq.get(new_seq, 0) + freq
        
    return new_word_freq


def build_word_freq_from_text(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    接收原始长文本和特殊字符列表，安全地进行预分词，并转换为底层的字节频次字典。
    """
    word_freq = Counter()
    
    # -------------------------------------------------------------
    # 步骤 1：使用“大砍刀”切分特殊 Token
    # -------------------------------------------------------------
    if special_tokens:
        # re.escape 的作用是把 "<|endoftext|>" 里的 "|" 等符号转义，防止正则解析错误
        # 加括号 f"({...})" 的目的是让 re.split 在切分文本时，把分隔符（也就是特殊 Token 本身）也保留在结果列表里
        escaped_tokens = [re.escape(t) for t in special_tokens]
        split_pattern = f"({'|'.join(escaped_tokens)})"
        
        # 比如文本是 "hello<|endoftext|>world"
        # chunks 会变成 ['hello', '<|endoftext|>', 'world']
        chunks = re.split(split_pattern, text)
    else:
        chunks = [text]

    # -------------------------------------------------------------
    # 步骤 2：精细化处理每一个文本块
    # -------------------------------------------------------------
    for chunk in chunks:
        # 过滤掉 re.split 可能产生的空字符串
        if not chunk:
            continue
            
        # 【分支 A】：如果这个块是特殊 Token，启动“绝对防御”
        if chunk in special_tokens:
            # 直接将整个特殊 Token 作为一个单独的整体存入
            # 💡 注意：这里没有按字节拆散！(b"<|endoftext|>",) 是一个只含有一个元素的元组
            token_bytes = chunk.encode("utf-8")
            word_freq[(token_bytes,)] += 1
            
        # 【分支 B】：如果是普通文本，上 GPT-2 正则“手术刀”
        else:
            # finditer 会找出所有匹配 GPT2_PAT 的单词块
            for match in re.finditer(GPT2_PAT, chunk):
                word = match.group()
                
                # 👑 核心转换：把字符串变成 UTF-8 字节，再拆成单字节的元组
                # 比如 "the" -> b"the" -> (b"t", b"h", b"e")
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                
                # 频次加一
                word_freq[word_bytes] += 1
                
    return dict(word_freq)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    BPE 训练主函数：读取语料，执行预分词，循环合并最高频字节对，直到达到目标词表大小。
    """
    # -------------------------------------------------------------
    # 阶段一：准备工作 (Initialization & Pre-tokenization)
    # -------------------------------------------------------------
    
    # 1. 初始化基础词表 (0-255 的单字节)
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # 2. 注册特殊 Token
    # 特殊 Token 必须作为一个完整的字节串存入词表，绝不能被拆散
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1
            
    # 3. 读取语料 (这里假设你已经写好了一个 read_corpus 和 pretokenize 的辅助函数)
    # 预分词一定要注意：先按 special_tokens 把文本切开，再对普通文本用正则切词
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # 🚨 注意：这里省略了具体的正则预分词实现细节。
    # 你需要把 text 切割，并转换成我们需要的 word_freq 字典格式。
    # 比如: word_freq = {(b"t", b"h", b"e"): 100, (b"<|endoftext|>",): 5, ...}
    word_freq = build_word_freq_from_text(text, special_tokens) 
    
    merges = []
    
    # 计算我们还需要进行多少次 merge 才能达到目标 vocab_size
    target_num_merges = vocab_size - len(vocab)
    
    # -------------------------------------------------------------
    # 阶段二：BPE 核心大循环 (The Big Loop)
    # -------------------------------------------------------------
    
    for _ in range(target_num_merges):
        # 1. 调用砖块二：统计当前所有的字节对频次
        pair_counts = get_pair_counts(word_freq)
        
        # 如果已经没有相邻的对可以合并了（比如所有词都变成了单体），提前结束
        if not pair_counts:
            break
            
        # 2. 👑 核心重难点：寻找最高频的 Pair，并处理平局 (Tie-break)
        # Python 的 max 函数非常强大：
        # key=lambda p: (pair_counts[p], p) 意味着它会先比较频次；
        # 如果频次相同，它会自动比较 p (也就是元组本身) 的字典序，选出更大的！
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        
        # 3. 记录这次合并
        merges.append(best_pair)
        
        # 4. 把合并出的新 token 加入词表
        # 注意：在 Python 里，b"t" + b"h" 会自动变成 b"th"
        merged_token_bytes = best_pair[0] + best_pair[1]
        vocab[next_id] = merged_token_bytes
        next_id += 1
        
        # 5. 调用砖块三：全局刷新词频字典，把所有的 best_pair 粘起来
        word_freq = apply_merge(word_freq, best_pair)

    # 训练结束，交卷！
    return vocab, merges


# Default public BPE entry points should use the optimized implementation.
from .tokenizer_optimized import Tokenizer as Tokenizer  # noqa: E402
from .tokenizer_optimized import train_bpe as train_bpe  # noqa: E402



'''
 =========测试代码块==========
if __name__ == "__main__":
    # 模拟我们最初统计出的词频字典
    current_word_freq = {
        (b"t", b"h", b"e"): 10,
        (b"t", b"h", b"a", b"t"): 5,
        (b"c", b"a", b"t"): 3
    }
    print("【初始状态】:", current_word_freq)

    # 1. 找出所有 pair 的频次 (砖块二)
    pair_counts = get_pair_counts(current_word_freq)
    print("\n【统计 Pair 频次】:", pair_counts)
    
    # 假设我们挑出了最高频的 (b't', b'h')，它出现了 15 次
    best_pair = (b"t", b"h")
    print(f"\n【决定合并最高频的 Pair】: {best_pair}")

    # 2. 全局应用合并，刷新字典 (砖块三，内部调用了砖块一)
    current_word_freq = apply_merge(current_word_freq, best_pair)
    print("\n【合并后的新字典】:", current_word_freq)
    # 预期输出中，所有的 (b't', b'h') 都变成了 b'th'，比如 (b'th', b'e'): 10
    

'''
