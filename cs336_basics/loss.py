import math
import torch
import torch.nn as nn


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算数值稳定的交叉熵损失。
    logits: 形状 (..., vocab_size)，包含模型预测的原始得分
    targets: 形状 (...)，包含真实的下一个词的 Token ID
    """
    # 1. 获取词表大小
    vocab_size = logits.size(-1)
    
    # 2. 扁平化张量 (C++ 视角的内存连续化处理)
    # 因为传进来的可能是 (batch_size, seq_len, vocab_size) 等多维张量
    # 我们统一 view 成 2D 的 (N, vocab_size)，方便做矩阵操作
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # 3. 提取正确词对应的原始得分 o_x
    # 使用 torch.gather 进行高级索引，相当于 C 数组里的 logits_flat[i][targets_flat[i]]
    # targets_flat.unsqueeze(-1) 是为了把形状从 (N) 变成 (N, 1) 匹配维度
    target_logits = torch.gather(logits_flat, dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
    
    # 4. 减最大值技巧 (计算 m)
    m = logits_flat.max(dim=-1, keepdim=True)[0]
    
    # 5. 计算 LogSumExp 项: m + log(sum(exp(logits - m)))
    # 注意这里内部的 sum 降维后，为了和 m 相加，也需要 squeeze
    exp_shifted = torch.exp(logits_flat - m)
    lse = m.squeeze(-1) + torch.log(torch.sum(exp_shifted, dim=-1))
    
    # 6. 计算每个样本的损失: -o_x + LSE
    loss_per_token = -target_logits + lse
    
    # 7. 根据作业要求，返回 batch 的平均损失
    return loss_per_token.mean()