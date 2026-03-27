import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        构造一个无偏置 (bias-free) 的线性层。
        """
        # 必须首先调用父类的构造函数，这是 PyTorch 的规矩，否则无法注册参数
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. 申请底层内存空间
        # 我们需要一个形状为 (out_features, in_features) 的张量
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_tensor = torch.empty((out_features, in_features), **factory_kwargs)
        
        # 2. 注入“灵魂”，将其升级为可学习的参数，并且按照作业要求命名为 W (而不是 WT) [cite: 540]
        self.weight = nn.Parameter(weight_tensor)
        
        # 3. 执行参数初始化 (👑 极其重要的考点)
        self._init_weights()

    def _init_weights(self):
        """
        使用截断正态分布 (Truncated Normal) 初始化权重。
        """
        # 计算标准差：方差为 2 / (in_features + out_features) 
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        
        # 截断在 [-3σ, 3σ] 的范围内 
        # trunc_normal_ 底层会直接修改 self.weight 的那块内存
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=std, 
            a=-3.0 * std, 
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：执行 y = x @ W^T
        x 的形状通常是 (batch_size, seq_len, in_features)
        """
        # self.weight 的形状是 (out_features, in_features)
        # 所以我们必须对其转置 (也就是 .T) 才能做矩阵乘法
        # 在 PyTorch 中，@ 符号等价于 torch.matmul
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        构造一个词嵌入层 (Embedding Layer)。
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 申请内存：“出版一本空白的特征字典”
        # 形状是 (词表大小, 每个词的特征维度)
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_tensor = torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        
        # 2. 注入灵魂：将其注册为可学习参数
        self.weight = nn.Parameter(weight_tensor)
        
        # 3. 执行作业要求的截断正态分布初始化
        self._init_weights()

    def _init_weights(self):
        """
        使用 N(0, 1) 且截断在 [-3, 3] 的正态分布初始化权重。
        """
        # 严格按照作业 3.4.1 的要求调用底层 C++ 实现的初始化函数
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=1.0, 
            a=-3.0, 
            b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播：极其暴力的 O(1) 内存切片寻址。
        token_ids 形状通常是 (batch_size, seq_len)
        """
        # PyTorch 的底层重载了 [] 运算符
        # 它会根据 token_ids 里的整数，直接去 self.weight 这块内存里把对应的行拷贝出来拼好
        # 输出的形状将是 (batch_size, seq_len, embedding_dim)
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        均方根归一化层。
        """
        super().__init__()
        self.eps = eps
        
        # 申请内存并注册为可学习参数 (这就是公式里的 g_i)
        # 根据作业要求，初始化为全 1
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：极其关键的类型转换与数学计算。
        x 的形状通常是 (batch_size, seq_len, d_model)
        """
        # 1. 保存输入原本的数据类型 (比如 float16 或 bfloat16)
        original_dtype = x.dtype
        
        # 2. Upcast 到 float32 进行高精度内部计算 (防止平方时数值溢出)
        x_fp32 = x.to(torch.float32)
        
        # 3. 计算 RMS (均方根)
        # x_fp32 ** 2: 求平方
        # torch.mean(..., dim=-1, keepdim=True): 在最后一个维度 (d_model) 上求均值
        # keepdim=True 极其重要，它能保证形状是 (batch, seq_len, 1)，这样才能和原来的 x 做广播除法
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 4. 归一化，并立刻 cast 回原来的数据类型
        x_normed = (x_fp32 / rms).to(original_dtype)
        
        # 5. 乘以可学习的增益参数 weight
        return x_normed * self.weight


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        
        # 👑 核心逻辑：计算 d_ff 并强制对齐到 64 的倍数
        # 按照作业要求：d_ff = (8/3) * d_model
        hidden_dim = int(8.0 * d_model / 3.0)
        
        # 向上/向下取整到最接近的 64 的倍数
        # 常用位运算或者数学技巧，这里使用标准数学写法确保清晰
        # 比如：(hidden_dim + 63) // 64 * 64
        # 为了保证与各种开源库对齐，这里我们加上取倍数逻辑
        MULTIPLE_OF = 64
        hidden_dim = MULTIPLE_OF * ((hidden_dim + MULTIPLE_OF - 1) // MULTIPLE_OF)
        
        # 我们需要实例化 3 个 Linear 层 (复用你手写的 Linear)
        # 对应公式里的 W1, W2, W3
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 直接使用传进来的 d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：执行 FFN(x) = W2( SiLU(W1(x)) * W3(x) )
        """
        # 分支 1：计算 W1(x) 并过 SiLU 激活函数
        # 使用 PyTorch 底层的 silu 算子
        branch1 = torch.nn.functional.silu(self.w1(x))
        
        # 分支 2：计算 W3(x)
        branch2 = self.w3(x)
        
        # 门控相乘后，通过 W2 映射回原维度
        return self.w2(branch1 * branch2)


class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.w1 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 缓存表。
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even d_k")
        
        # 1. 预先计算每对维度的频率 (1 / Theta^(2k/d))
        # 我们需要生成一系列角度，从 0 步进到 d_k，步长为 2
        # 注意公式里的 k 是从 1 开始的，转换为索引 0, 2, 4 ...
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        
        # 2. 生成绝对位置的序列 i = 0, 1, 2, ... max_seq_len - 1
        seq_positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        
        # 3. 利用外积 (Outer Product) 生成所有的角度 theta_{i, k}
        # 形状: (max_seq_len, d_k // 2)
        angles = torch.outer(seq_positions, inv_freq)
        
        # 4. 每个旋转角对应一个 2x2 块，所以需要把每个角重复两次：
        # [theta_1, theta_2] -> [theta_1, theta_1, theta_2, theta_2]
        angles = angles.repeat_interleave(2, dim=-1)
        
        # 5. 提前把 cos 和 sin 算好，注册为 buffer (不需要梯度的持久化状态)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将预计算的 RoPE 应用到输入张量 x 上。
        x 的形状通常是 (batch_size, seq_len, num_heads, d_k) 或类似形状
        token_positions 形状是 (batch_size, seq_len)
        """
        d_k = x.shape[-1]
        if d_k % 2 != 0:
            raise ValueError("RoPE requires the last dimension of x to be even")

        # 1. 按偶/奇维成对旋转，而不是把向量直接切成前后两半
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        
        # 3. 根据 token_positions，去缓存的表里提取对应的 cos 和 sin 值
        # cos_cached 的形状是 (max_seq_len, d_k)
        # 经过高级索引后，cos 和 sin 的形状将变为 (batch_size, seq_len, d_k)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 4. 把 cos/sin reshape 成能和 x 广播的形状。
        # 这里约定 x 的最后一个非特征维是 sequence 维，前面可以有任意 batch/head 维。
        # 例如：
        # - x: (batch, seq, d_k), token_positions: (seq,)      -> (1, seq, d_k)
        # - x: (batch, seq, d_k), token_positions: (batch, seq)-> (batch, seq, d_k)
        # - x: (batch, heads, seq, d_k), token_positions: (1, seq) -> (1, 1, seq, d_k)
        prefix_shape = x.shape[:-1]
        token_shape = tuple(token_positions.shape)

        if not token_shape:
            raise ValueError("token_positions must have at least one dimension")
        if token_shape[-1] != prefix_shape[-1]:
            raise ValueError("The last dimension of token_positions must match the sequence length of x")

        leading_token_shape = token_shape[:-1]
        leading_x_shape = prefix_shape[: len(leading_token_shape)]
        for token_dim, x_dim in zip(leading_token_shape, leading_x_shape):
            if token_dim not in (1, x_dim):
                raise ValueError("Leading token_positions dimensions must be broadcast-compatible with x")

        expand_shape = (
            leading_token_shape
            + (1,) * (len(prefix_shape) - len(token_shape))
            + (token_shape[-1], d_k)
        )

        cos = cos.reshape(expand_shape)
        sin = sin.reshape(expand_shape)
        
        # 5. 应用公式: x * cos + [-x2, x1] * sin
        return (x * cos) + (x_rotated * sin)
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    数值稳定的 Softmax 实现。
    """
    # 1. 找到指定维度上的最大值，keepdim=True 保证能进行广播减法
    # x.max() 会返回 (values, indices)，我们只需要 values
    x_max = x.max(dim=dim, keepdim=True)[0]
    
    # 2. 所有元素减去最大值 (极其关键的安全护城河)
    x_safe = x - x_max
    
    # 3. 计算 exp
    exp_x = torch.exp(x_safe)
    
    # 4. 除以 exp 的总和，得到概率分布
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    计算缩放点积注意力。
    q, k 形状: (batch_size, ..., seq_len, d_k)
    v 形状: (batch_size, ..., seq_len, d_v)
    mask 形状: (seq_len, seq_len) 布尔值，True 允许关注，False 屏蔽
    """
    # 1. 获取 d_k，用于缩放
    d_k = q.size(-1)
    
    # 2. 计算 Q * K^T
    # 因为有 batch 和 多头维度，我们不能简单用 .T
    # 我们只对最后两个维度 (seq_len, d_k) 进行转置，变成 (d_k, seq_len)
    # 使用 .transpose(-2, -1) 来实现
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # 3. 缩放 (除以 sqrt(d_k))
    scores = scores / math.sqrt(d_k)
    
    # 4. 应用 Mask (如果有的话)
    if mask is not None:
        # mask 中为 False 的地方，替换为极小的负数 (比如 -1e9)
        # 这样在 softmax 后，这些地方的概率就变成了 0
        scores = scores.masked_fill(~mask, -1e9)
        
    # 5. 计算 softmax 概率 (在我们刚刚写的 softmax 维度上，通常是最后一个维度)
    attention_weights = softmax(scores, dim=-1)
    
    # 6. 乘以 V 得到最终输出
    output = torch.matmul(attention_weights, v)
    
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        rope_theta: float | None = None,
        device=None,
        dtype=None,
    ):
        """
        因果多头自注意力机制。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 强制整除，计算每个头的维度
        # 根据作业要求: d_k = d_v = d_model / h
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = d_model // num_heads
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 1. 实例化 4 个你手搓的 Linear 模块 (全都不带 bias)
        # 分别用于生成 Q, K, V 以及最后的输出投影 O
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.out_proj = Linear(d_model, d_model, **factory_kwargs)
        
        self.rope = None
        if max_seq_len is not None and rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播。x 形状: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 经过线性层，得到 Q, K, V
        # 此时形状依然是 (batch_size, seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 👑 张量切分与轴置换 (GPU 并行的核心魔法)
        # view: 把 d_model 拆成 (num_heads, d_k)
        # transpose(1, 2): 把 num_heads 提到前面，变成 (batch_size, num_heads, seq_len, d_k)
        # 这样后面算点积时，就会在 seq_len 和 d_k 维度上进行矩阵相乘，而在 batch 和 head 上并行！
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # 3. 生成因果掩码 (Causal Mask)
        # torch.ones 生成全 1 矩阵，torch.tril 取它的下三角部分 (对角线以上变 0)
        # 最后转成布尔值: True 表示允许关注，False 表示屏蔽
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        
        # 4. 调用我们刚刚写好的单头注意力算子
        # 它的底层广播机制会完美处理掉多出来的 batch 和 num_heads 维度
        # 输出形状: (batch_size, num_heads, seq_len, d_k)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 5. 把多头拼接回去 (Concat)
        # 先转置回来: (batch_size, seq_len, num_heads, d_k)
        # contiguous(): 因为转置打乱了底层的内存连续性，必须重新申请连续内存，才能用 view
        # view: 重新把最后两个维度合并回 d_model
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 6. 最后过一遍输出投影 W_O
        return self.out_proj(attn_out)
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        rope_theta: float | None = None,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        ffn_style: str = "swiglu",
        device=None,
        dtype=None,
    ):
        """
        组装单个可切换 Pre-norm / Post-norm 的 Transformer Block。
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        norm_cls = RMSNorm if use_rmsnorm else IdentityNorm
        if norm_style not in {"pre", "post"}:
            raise ValueError(f"Unsupported norm_style: {norm_style}")
        if ffn_style not in {"swiglu", "silu"}:
            raise ValueError(f"Unsupported ffn_style: {ffn_style}")
        self.norm_style = norm_style
        self.ffn_style = ffn_style
        ffn_cls = SwiGLU if ffn_style == "swiglu" else SiLUFFN
        
        # 实例化咱们手搓的四个大件！
        self.norm1 = norm_cls(d_model, **factory_kwargs) if use_rmsnorm else norm_cls()
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            **factory_kwargs,
        )
        self.norm2 = norm_cls(d_model, **factory_kwargs) if use_rmsnorm else norm_cls()
        self.ffn = ffn_cls(d_model, d_ff, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播：极其干净的残差连接。
        """
        if self.norm_style == "pre":
            x = x + self.attn(self.norm1(x), token_positions=token_positions)
            x = x + self.ffn(self.norm2(x))
            return x

        z = self.norm1(x + self.attn(x, token_positions=token_positions))
        y = self.norm2(z + self.ffn(z))
        return y
    
class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        rope_theta: float | None = None,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        ffn_style: str = "swiglu",
        device=None, 
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        norm_cls = RMSNorm if use_rmsnorm else IdentityNorm
        if norm_style not in {"pre", "post"}:
            raise ValueError(f"Unsupported norm_style: {norm_style}")
        if ffn_style not in {"swiglu", "silu"}:
            raise ValueError(f"Unsupported ffn_style: {ffn_style}")
        self.norm_style = norm_style
        self.ffn_style = ffn_style
        
        # 1. 大门：Embedding 层，把数字 ID 变成稠密向量
        self.embedding = Embedding(vocab_size, d_model, **factory_kwargs)
        
        # 2. 核心引擎组：用 nn.ModuleList 把 num_layers 个 Block 串联起来
        # 这就像 C++ 里的 std::vector<TransformerBlock*>
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                use_rmsnorm=use_rmsnorm,
                norm_style=norm_style,
                ffn_style=ffn_style,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        ])
        
        # 3. 终极裁判：最后再做一次 RMSNorm
        self.final_norm = norm_cls(d_model, **factory_kwargs) if use_rmsnorm else norm_cls()
        
        # 4. 翻译官：LM Head (无偏置线性层)
        # 它的任务是把 d_model 维的特征，映射回 vocab_size 维，用来预测下一个词！
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: 形状 (batch_size, seq_len) 的整数张量
        返回 logits: 形状 (batch_size, seq_len, vocab_size)
        """
        # 1. 查表，得到初始特征 x
        x = self.embedding(token_ids)
        token_positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        
        # 2. 像流水线一样，依次穿过所有的 Transformer Block
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
            
        # 3. 最终归一化
        x = self.final_norm(x)
        
        # 4. 映射到词表概率空间 (Logits)
        logits = self.lm_head(x)
        
        return logits
