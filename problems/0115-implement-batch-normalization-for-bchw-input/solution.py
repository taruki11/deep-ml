import numpy as np
from typing import Tuple, Optional

def batch_norm(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: Optional[np.ndarray] = None,
    running_var: Optional[np.ndarray] = None,
    training: bool = True,
    momentum: float = 0.1,
    epsilon: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 BCHW 格式的 4D 特征图执行 Batch Normalization。
    
    参数:
        X: 4D 数组，形状为 (B, C, H, W)
        gamma: 缩放参数，形状为 (1, C, 1, 1) 或 (C,)
        beta: 平移参数，形状为 (1, C, 1, 1) 或 (C,)
        running_mean: 历史滑动均值
        running_var: 历史滑动方差
        training: True 表示训练模式，False 表示推理模式
        momentum: 滑动统计量更新动量，默认 0.1
        epsilon: 数值稳定常数，默认 1e-5
        
    返回:
        (output, running_mean, running_var)
    """
    X = np.asarray(X, dtype=float)
    C = X.shape[1]

    # 记录原始 running stats 维度（若传入 1D，返回时保持 1D）
    rm_orig_shape = running_mean.shape if running_mean is not None else None
    rv_orig_shape = running_var.shape if running_var is not None else None

    # 初始化 running_mean 与 running_var
    if running_mean is None:
        running_mean = np.zeros((1, C, 1, 1), dtype=float)
    else:
        running_mean = np.asarray(running_mean, dtype=float)
        if running_mean.ndim == 1:
            running_mean = running_mean.reshape(1, C, 1, 1)

    if running_var is None:
        running_var = np.ones((1, C, 1, 1), dtype=float)
    else:
        running_var = np.asarray(running_var, dtype=float)
        if running_var.ndim == 1:
            running_var = running_var.reshape(1, C, 1, 1)

    # 确保 gamma 与 beta 形状广播为 (1, C, 1, 1)
    gamma_res = np.asarray(gamma, dtype=float)
    beta_res = np.asarray(beta, dtype=float)
    if gamma_res.ndim == 1:
        gamma_res = gamma_res.reshape(1, C, 1, 1)
    if beta_res.ndim == 1:
        beta_res = beta_res.reshape(1, C, 1, 1)

    if training:
        # 在 (0, 2, 3) 维度（即 Batch, Height, Width）聚合计算均值与方差
        batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)

        # 归一化并进行仿射缩放和平移
        X_norm = (X - batch_mean) / np.sqrt(batch_var + epsilon)
        out = gamma_res * X_norm + beta_res

        # 动量更新历史统计量: running_stat = (1 - momentum) * running + momentum * batch
        running_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
        running_var = (1.0 - momentum) * running_var + momentum * batch_var
    else:
        # 推理模式：使用固化的历史统计量
        X_norm = (X - running_mean) / np.sqrt(running_var + epsilon)
        out = gamma_res * X_norm + beta_res

    # 还原初始传入的维度形态
    if rm_orig_shape is not None and len(rm_orig_shape) == 1:
        running_mean = running_mean.reshape(rm_orig_shape)
    if rv_orig_shape is not None and len(rv_orig_shape) == 1:
        running_var = running_var.reshape(rv_orig_shape)

    return out, running_mean, running_var

# 别名兼容
batch_normalization = batch_norm