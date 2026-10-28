import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    生成标准 Transformer 正弦位置编码表
    
    参数:
        seq_len: 序列长度 (例如 4)
        d_model: 隐藏层维度 (必须为偶数，例如 4)
    返回:
        pe: 形状为 (seq_len, d_model) 的位置张量
    """
    # 1. 预分配输出张量
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    
    # 2. 构造位置序列列向量: shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    
    # 3. 在对数空间计算各通道的角频率除数: 1 / 10000^(2i / d_model)
    # 形状为 (d_model // 2,)
    i_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(i_indices * (-math.log(10000.0) / d_model))
    
    # 4. 外积广播相乘，计算全部通道的角度: (seq_len, 1) * (d_model // 2,) -> (seq_len, d_model // 2)
    angles = position * div_term
    
    # 5. 偶数列填充 sin，奇数列填充 cos
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    
    return pe