import math

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    计算给定迭代步数下的学习率。
    """
    # 1. Warm-up 阶段: 线性增长
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
        
    # 2. Post-annealing 阶段: 超过余弦周期后，保持最小值
    if it >= cosine_cycle_iters:
        return min_learning_rate
        
    # 3. Cosine annealing 阶段: 余弦衰减
    # 计算当前在余弦周期内的进度比例 (0 到 1 之间)
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    # 使用余弦公式计算系数
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

import torch
from typing import Iterable

def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    """
    对一组参数的梯度进行原地裁剪。
    """
    # 提取所有包含梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
        
    # 1. 计算所有梯度的整体 L2 范数平方和
    total_norm_sq = 0.0
    for p in params_with_grad:
        # detach() 防止把这里的计算加入计算图
        total_norm_sq += p.grad.detach().pow(2).sum().item()
        
    # 开根号得到真实的 L2 范数
    total_norm = math.sqrt(total_norm_sq)
    
    # 2. 如果整体范数超过了最大阈值，进行等比例缩放
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.detach().mul_(scale)