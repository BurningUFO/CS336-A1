import os
import urllib.request
import gzip
import shutil

def main():
    # 确保 data 文件夹存在
    os.makedirs("data", exist_ok=True)

    # 需要下载的文件列表 (URL, 保存路径)
    downloads = [
        ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-train.txt"),
        ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt", "data/TinyStoriesV2-GPT4-valid.txt"),
        ("https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz", "data/owt_train.txt.gz"),
        ("https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz", "data/owt_valid.txt.gz")
    ]

    # 1. 下载文件
    for url, path in downloads:
        if not os.path.exists(path) and not os.path.exists(path.replace(".gz", "")):
            print(f"⏳ 正在下载 {path} ... 这可能需要几分钟，取决于网速。")
            urllib.request.urlretrieve(url, path)
            print(f"✅ {path} 下载完成！")
        else:
            print(f"⏩ {path} 已存在，跳过下载。")

    # 2. 解压 .gz 文件 (对应 Linux 的 gunzip)
    gz_files = [
        ("data/owt_train.txt.gz", "data/owt_train.txt"),
        ("data/owt_valid.txt.gz", "data/owt_valid.txt")
    ]
    
    for gz_path, txt_path in gz_files:
        if os.path.exists(gz_path) and not os.path.exists(txt_path):
            print(f"📦 正在解压 {gz_path} ...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(txt_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"✅ 解压完成: {txt_path}")

    print("🎉 所有数据准备完毕！可以开始训练啦！")

if __name__ == "__main__":
    main()