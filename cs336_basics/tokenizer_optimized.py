# 必须导入 regex 库（比 Python 自带的 re 库更强大，支持高级 Unicode 匹配）
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

import regex as re

# 这是 GPT-2 使用的经典正则表达式（来自 OpenAI 的 tiktoken [cite: 154]）
# 它的作用是聪明地把文本切碎，比如把 "i'll" 切成 "i" 和 "'ll"，把连续的空格切到一起。
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_RE = re.compile(GPT2_PAT)
SINGLE_BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

WordTokens = tuple[bytes, ...]
TokenPair = tuple[bytes, bytes]


class _ReverseLexPair:
    __slots__ = ("pair",)

    def __init__(self, pair: TokenPair):
        self.pair = pair

    def __lt__(self, other: "_ReverseLexPair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReverseLexPair) and self.pair == other.pair


def _iter_pairs(word_tokens: WordTokens):
    for i in range(len(word_tokens) - 1):
        yield (word_tokens[i], word_tokens[i + 1])


def _compile_special_token_re(special_tokens: list[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return re.compile(f"({'|'.join(escaped_tokens)})")


def _get_word_pair_counts(word_tokens: WordTokens) -> Counter[TokenPair]:
    return Counter(_iter_pairs(word_tokens))


def _add_word_to_pair_index(
    pair_to_words: dict[TokenPair, set[WordTokens]], word_tokens: WordTokens
) -> None:
    for pair in set(_iter_pairs(word_tokens)):
        pair_to_words[pair].add(word_tokens)


def _remove_word_from_pair_index(
    pair_to_words: dict[TokenPair, set[WordTokens]], word_tokens: WordTokens
) -> None:
    for pair in set(_iter_pairs(word_tokens)):
        words = pair_to_words.get(pair)
        if words is None:
            continue
        words.discard(word_tokens)
        if not words:
            del pair_to_words[pair]


def _push_pair_heap(
    heap: list[tuple[int, _ReverseLexPair, TokenPair]],
    pair: TokenPair,
    count: int,
) -> None:
    if count > 0:
        heapq.heappush(heap, (-count, _ReverseLexPair(pair), pair))


def _pop_best_pair(
    heap: list[tuple[int, _ReverseLexPair, TokenPair]],
    pair_counts: Counter[TokenPair],
) -> TokenPair | None:
    while heap:
        neg_count, _, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count <= 0:
            continue
        if -neg_count != count:
            continue
        return pair
    return None


def merge_word(word_tokens: WordTokens, pair: TokenPair) -> WordTokens:
    """
    在一个 token 序列里，把指定的 pair 从左到右进行贪心合并。
    """
    left, right = pair
    merged: list[bytes] = []
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


def get_pair_counts(word_freq: dict[WordTokens, int]) -> dict[TokenPair, int]:
    """
    遍历全局词频字典，统计所有相邻 token pair 的总频次。
    """
    pair_counts: Counter[TokenPair] = Counter()

    for token_seq, freq in word_freq.items():
        # 遍历当前单词的所有相邻对
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            # 累加这个单词在全局出现的频次
            pair_counts[pair] += freq

    return pair_counts


def apply_merge(
    word_freq: dict[WordTokens, int],
    pair: TokenPair,
) -> dict[WordTokens, int]:
    """
    遍历全局词频字典，把指定的 pair 全部合并，并返回一个全新的词频字典。
    """
    new_word_freq: dict[WordTokens, int] = {}

    for token_seq, freq in word_freq.items():
        # 生成合并后的新序列
        new_seq = merge_word(token_seq, pair)

        # 把新序列放进新字典，并累加频次
        # .get(new_seq, 0) 的意思是：如果字典里还没有这个 key，就默认返回 0
        new_word_freq[new_seq] = new_word_freq.get(new_seq, 0) + freq

    return new_word_freq


def build_word_freq_from_text(text: str, special_tokens: list[str]) -> dict[WordTokens, int]:
    """
    接收原始长文本和特殊字符列表，安全地进行预分词，并转换为底层的字节频次字典。
    """
    word_freq: Counter[WordTokens] = Counter()
    special_tokens_set = set(special_tokens)
    special_token_re = _compile_special_token_re(special_tokens)

    # -------------------------------------------------------------
    # 步骤 1：使用“大砍刀”切分特殊 Token
    # -------------------------------------------------------------
    if special_token_re is not None:
        # 比如文本是 "hello<|endoftext|>world"
        # chunks 会变成 ['hello', '<|endoftext|>', 'world']
        chunks = special_token_re.split(text)
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
        if chunk in special_tokens_set:
            # 直接将整个特殊 Token 作为一个单独的整体存入
            # 注意：这里没有按字节拆散！(b"<|endoftext|>",) 是一个只含有一个元素的元组
            token_bytes = chunk.encode("utf-8")
            word_freq[(token_bytes,)] += 1

        # 【分支 B】：如果是普通文本，上 GPT-2 正则“手术刀”
        else:
            # 提前编译正则，并重用 256 个单字节对象，减少预分词常数开销
            for match in GPT2_RE.finditer(chunk):
                word = match.group()
                word_bytes = tuple(SINGLE_BYTE_TOKENS[b] for b in word.encode("utf-8"))
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
    vocab = {i: SINGLE_BYTE_TOKENS[i] for i in range(256)}
    next_id = 256

    # 2. 注册特殊 Token
    # 特殊 Token 必须作为一个完整的字节串存入词表，绝不能被拆散
    vocab_values = set(vocab.values())
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab_values:
            vocab[next_id] = token_bytes
            vocab_values.add(token_bytes)
            next_id += 1

    # 3. 读取语料
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    word_freq = build_word_freq_from_text(text, special_tokens)

    merges: list[TokenPair] = []

    # 计算我们还需要进行多少次 merge 才能达到目标 vocab_size
    target_num_merges = vocab_size - len(vocab)

    # -------------------------------------------------------------
    # 阶段二：增量维护 pair 频次，而不是每轮全量重扫
    # -------------------------------------------------------------

    pair_counts: Counter[TokenPair] = Counter()
    pair_to_words: dict[TokenPair, set[WordTokens]] = defaultdict(set)
    pair_heap: list[tuple[int, _ReverseLexPair, TokenPair]] = []

    for token_seq, freq in word_freq.items():
        word_pair_counts = _get_word_pair_counts(token_seq)
        for pair, count in word_pair_counts.items():
            pair_counts[pair] += count * freq
        _add_word_to_pair_index(pair_to_words, token_seq)

    for pair, count in pair_counts.items():
        _push_pair_heap(pair_heap, pair, count)

    for _ in range(target_num_merges):
        best_pair = _pop_best_pair(pair_heap, pair_counts)
        if best_pair is None:
            break

        merges.append(best_pair)

        merged_token_bytes = best_pair[0] + best_pair[1]
        vocab[next_id] = merged_token_bytes
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

    return vocab, merges


class Tokenizer:
    '''
    BPE 编解码器类。负责将文本编码为整数 ID 列表，以及将整数 ID 列表解码回文本。
    '''
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[TokenPair], 
        special_tokens: list[str] | None = None
    ):
        """
        构造函数。接收训练好的词表、合并规则和特殊字符列表，并将其存为类的内部状态。
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens_set = set(self.special_tokens)
        self.special_token_re = _compile_special_token_re(self.special_tokens)
        
        # 💡 核心优化：建立反向词表 (bytes -> int)，将 encode 查表时间复杂度降为 O(1)
        self.inverse_vocab = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        
        # 预先把 special_tokens 转换成 bytes 存起来，方便后续使用
        self.special_token_bytes = {t.encode("utf-8") for t in self.special_tokens}
        
    def encode(self, text: str) -> list[int]:
        """
        将文本字符串编码为 Token ID 列表。
        """
        # 第一道工序：保序预分词
        # 1. 先用特殊字符把长文本切开
        if self.special_token_re is not None:
            raw_chunks = self.special_token_re.split(text)
        else:
            raw_chunks = [text]

        # ordered_word_tokens 用来按顺序存放切好的词块
        # 比如 "the cat" 会变成 [(b't', b'h', b'e'), (b' ', b'c', b'a', b't')]
        ordered_word_tokens: list[tuple[bytes, ...]] = []
        
        for chunk in raw_chunks:
            if not chunk:
                continue
            # 如果是特殊 Token，绝对不能切碎，作为一个整体存入
            if chunk in self.special_tokens_set:
                ordered_word_tokens.append((chunk.encode("utf-8"),))
            else:
                # 如果是普通文本，用正则手术刀切成单词，再转成单字节元组
                for match in GPT2_RE.finditer(chunk):
                    word = match.group()
                    word_bytes = tuple(SINGLE_BYTE_TOKENS[b] for b in word.encode("utf-8"))
                    ordered_word_tokens.append(word_bytes)
        
        # 第二道工序：严格按规则合并
        final_ids: list[int] = []
        
        for tokens in ordered_word_tokens:
            # 用一个变量暂存当前词块的状态，方便我们在上面不断进行合并
            current_tokens = tokens
            
            # 只要词块长度大于 1，说明里面还有相邻对，就继续尝试合并
            while len(current_tokens) > 1:
                # 获取当前词块里【所有的】相邻对
                # 比如 (b"t", b"h", b"e") 会得到 {(b"t", b"h"), (b"h", b"e")}
                current_pairs = set(_iter_pairs(current_tokens))
                
                # 👑 核心逻辑：遍历我们的官方规则表，找出优先级最高（最靠前）的规则
                applicable_rule = None
                for rule_pair in self.merges:
                    if rule_pair in current_pairs:
                        applicable_rule = rule_pair
                        break  # 找到了！立刻跳出 for 循环，这就是我们要合并的对象
                
                # 如果遍历完整个规则表，都没发现可以适用的规则，说明这个词块已经合并到极限了
                if applicable_rule is None:
                    break
                    
                # 找到了适用的规则，调用你之前写好的神仙函数 merge_word 进行粘合！
                current_tokens = merge_word(current_tokens, applicable_rule)

        # 第三道工序：查表变 ID
            for token_bytes in current_tokens:
                # 在 O(1) 的时间复杂度内，瞬间把字节序列变成整数 ID！
                final_ids.append(self.inverse_vocab[token_bytes])
                
        return final_ids

    def decode(self, ids: list[int]) -> str:
        """
        将 Token ID 序列还原为原始字符串。
        """
        decoded_bytes = bytearray()
        for token_id in ids:
            decoded_bytes.extend(self.vocab[token_id])
        # 单个 token 可能只包含一个多字节 UTF-8 序列的前缀；对这种情况返回 replacement
        # character 比直接抛异常更符合 tokenizer.decode([id]) 的使用方式。
        return decoded_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        接收一个字符串的可迭代对象(比如文件句柄),惰性地(lazily)逐个 yield 出 token ID。
        这对于处理内存装不下的大文件至关重要。
        """
        # 遍历外面的每一块文本（比如文件的每一行）
        for text_chunk in iterable:
            # 遇到空字符串直接跳过
            if not text_chunk:
                continue
                
            # 直接白嫖我们刚才辛辛苦苦写好的 encode 核心引擎
            ids = self.encode(text_chunk)
            
            # 用 yield 惰性地一个一个把 ID 吐出去
            for token_id in ids:
                yield token_id
    
