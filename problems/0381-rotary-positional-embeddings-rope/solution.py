import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    PyTorch 纯张量化 RoPE 实现
    
    参数:
        x: 形状 (seq_len, d)，要求 d 为偶数
        positions: 形状 (seq_len,) 的 1D 整数/浮点张量
        base: 旋转频率底数 (默认 10000.0)
    """
    seq_len, d = x.shape
    d_half = d // 2
    device = x.device
    dtype = x.dtype

    # 1. 频率计算 theta_i = 1.0 / (base ** (2i / d))
    # 保证在相同 device 上以 float32 计算角度，避免半精度 (fp16) 下下溢
    i = torch.arange(d_half, device=device, dtype=torch.float32)
    theta = 1.0 / (base ** (2.0 * i / d))  # 形状: (d_half,)

    # 2. 计算角度矩阵: (seq_len, 1) * (1, d_half) -> (seq_len, d_half)
    angles = positions.unsqueeze(-1).float() * theta.unsqueeze(0)
    
    # 转回输入张量的数据类型 (如 bfloat16 / float32)
    cos = torch.cos(angles).to(dtype)  # (seq_len, d_half)
    sin = torch.sin(angles).to(dtype)  # (seq_len, d_half)

    # 3. 提取连续的偶数、奇数分量
    x_even = x[:, 0::2]  # (seq_len, d_half)
    x_odd  = x[:, 1::2]  # (seq_len, d_half)

    # 4. 2D 旋转矩阵乘法
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    # 5. 优雅拼装：stack 在最后一维变成 (seq_len, d_half, 2)，再展平为 (seq_len, d)
    # 完全避免了 in-place 切片赋值，原生支持 Autograd 反向传播
    out = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(start_dim=-2)
    
    return out

# --- 验证测试 ---
if __name__ == "__main__":
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    pos = torch.tensor([0, 1])
    out = apply_rope(x, pos)
    print("输出张量:\n", out)
    # 输出:
    # tensor([[ 1.0000,  0.0000],
    #         [-0.8415,  0.5403]])