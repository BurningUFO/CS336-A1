import time
import os
import pickle
from cs336_basics.tokenizer_optimized import train_bpe

def main():
    # 1. 设置好训练参数（与作业文档要求一致）
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # 检查数据文件是否存在 (之前我们在 README 里用 wget 下载过)
    if not os.path.exists(input_path):
        print(f"❌ 找不到数据文件：{input_path}")
        print("请确保你已经在终端里运行过 README 中的 mkdir 和 wget 下载命令！")
        return

    print(f"🚀 开始在 {input_path} 上训练 BPE...")
    print(f"🎯 目标词表大小: {vocab_size}")
    print("⏳ 这可能需要几十分钟，请耐心等待...")

    # 2. 点火！记录时间
    start_time = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    end_time = time.time()

    print(f"\n✅ 训练完成！总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    print(f"📊 实际生成的词表大小: {len(vocab)}")

    # 3. 👑 回答作业的书面问题：找出最长的 Token
    # max() 函数搭配 key=len，可以直接找出字典 values 中长度最长的 bytes 对象
    longest_token_bytes = max(vocab.values(), key=len)
    
    # 尝试把它解码成人类可读的字符串
    try:
        longest_token_str = longest_token_bytes.decode("utf-8")
        print(f"🏆 词表中最长的 Token 是: '{longest_token_str}'")
    except UnicodeDecodeError:
        # 如果包含无法直接打印的半截字节，就打印它的 bytes 原貌
        print(f"🏆 词表中最长的 Token (原始 bytes) 是: {longest_token_bytes}")
        
    print(f"📏 它的字节长度是: {len(longest_token_bytes)}")

    # 4. 保存我们的“胜利果实”（为后续的 Tokenizer 和语言模型做准备）
    # 在 Python 里，保存带有 bytes 对象的复杂字典，用 pickle 序列化是最简单省事的
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("checkpoints/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
        
    print("\n💾 词表和合并规则已成功保存至 checkpoints/ 目录下！")

if __name__ == "__main__":
    main()