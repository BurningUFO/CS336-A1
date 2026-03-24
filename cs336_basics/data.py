import os

import numpy as np
import torch

def get_batch(
    tokens: np.ndarray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维的 Token 数组中随机抽取一个 batch 的输入 (x) 和目标 (y)。
    """
    # 1. 计算合法起点的数量。
    # 最后一个合法起点是 len(tokens) - context_length - 1，而 randint 的 high 是开区间，
    # 所以这里应传入 len(tokens) - context_length。
    num_possible_start_indices = len(tokens) - context_length

    # 2. 随机生成 batch_size 个起始位置
    start_indices = torch.randint(0, num_possible_start_indices, (batch_size,)).tolist()
    
    # 3. 根据起始位置进行切片，并转换为张量
    # 注意：为了性能，通常先把 numpy 数组切片，转成张量，再送到 GPU (device)
    x_list = []
    y_list = []
    
    for i in start_indices:
        # 输入 x: 从 i 到 i + context_length
        x_chunk = tokens[i : i + context_length]
        # 目标 y: 从 i + 1 到 i + context_length + 1 (错开一位)
        y_chunk = tokens[i + 1 : i + context_length + 1]
        
        # 统一转为 int64 (torch.long) 的张量，这是 embedding 层和 cross_entropy 默认要求的类型
        x_list.append(torch.tensor(x_chunk, dtype=torch.long))
        y_list.append(torch.tensor(y_chunk, dtype=torch.long))
        
    # 4. 把列表堆叠成二维张量，并推送到指定的设备 (GPU/CPU)
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)

    return x, y

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike):
    """
    将模型、优化器状态和当前步数保存到硬盘。
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    从硬盘读取存档，恢复模型和优化器，并返回当前步数。
    """
    # 极其重要的底层优化：map_location="cpu"
    # 如果存档是在 GPU 上保存的，直接 load 会默认加载到原本的 GPU 上，可能导致显存瞬间爆满 (OOM)
    # 先加载到 CPU 的内存里，再分别让 model 和 optimizer 去吸收它们，是最安全的做法
    checkpoint = torch.load(src, map_location="cpu")
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"]
