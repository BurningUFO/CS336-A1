import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        初始化 AdamW 优化器。
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        # defaults 是给父类管理超参数的字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        执行单一优化步。这会在每次反向传播 (backward) 之后被调用。
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 遍历所有的参数组 (通常只有一个，除非你给不同网络层设置了不同的学习率)
        for group in self.param_groups:
            # 提取超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                # 如果这个参数没有梯度，说明它没参与计算，跳过
                if p.grad is None:
                    continue
                
                # 获取当前梯度 g
                grad = p.grad.data
                
                # 获取这个参数的状态字典 (第一次访问时为空)
                state = self.state[p]

                # 状态初始化 (相当于 C++ 里的懒加载)
                if len(state) == 0:
                    state['step'] = 0
                    # 初始化一阶和二阶矩，形状必须和参数 p 一模一样
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # 👑 开始执行 Algorithm 1 的硬核更新逻辑
                
                # 1. 更新一阶矩 m: m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                
                # 2. 更新二阶矩 v: v = beta2 * v + (1 - beta2) * (grad * grad)
                # addcmul_ 是一步到位的底层算子：加上 (张量1 * 张量2 * value)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. 计算偏置修正后的学习率 alpha_t
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 4. 更新参数: theta = theta - alpha_t * m / (sqrt(v) + eps)
                # 同样使用原地操作 addcdiv_ 以节省内存
                p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)

                # 5. 解耦的权重衰减 (Weight Decay)
                if weight_decay > 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
