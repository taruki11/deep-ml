import torch

def global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 全局平均池化:
    - 若输入为 (H, W, C): 沿 dim=(0, 1) 求均值，输出 (C,)
    - 若输入为 (B, C, H, W): 沿空间维度 dim=(-2, -1) 求均值，输出 (B, C)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)

    if x.ndim == 3:
        # (H, W, C) 格式
        return torch.mean(x, dim=(0, 1))
    elif x.ndim == 4:
        # 工业界标准 (B, C, H, W) 格式
        return torch.mean(x, dim=(-2, -1))
    else:
        raise ValueError(f"预期 3D 或 4D 张量，收到形状: {x.shape}")