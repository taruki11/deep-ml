import numpy as np
from typing import Any

def clip_gradients_by_global_norm(gradients: Any, max_norm: float):
    """
    根据全局 L2 范数对梯度进行裁剪 (Gradient Clipping by Global Norm)。
    
    参数:
        gradients: 输入梯度，支持:
                   - 单个 np.ndarray
                   - 梯度列表 (例如 [[3.0, 4.0], [0.0, 0.0]] 或 [np.array(...), np.array(...)])
        max_norm (float): 允许的最大全局 L2 范数阈值。
        
    返回:
        保持原始结构不变的裁剪后梯度。
    """
    # 1. 单个 NumPy 数组输入
    if isinstance(gradients, np.ndarray):
        norm = float(np.linalg.norm(gradients))
        if norm <= max_norm or norm == 0.0:
            return gradients.astype(float)
        return (gradients * (max_norm / norm)).astype(float)

    # 2. 梯度列表输入：汇总所有参数梯度的二范数平方和
    total_sq_norm = sum(np.sum(np.square(np.asarray(g, dtype=float))) for g in gradients)
    global_norm = float(np.sqrt(total_sq_norm))

    # 计算缩放系数
    if global_norm <= max_norm or global_norm == 0.0:
        scale = 1.0
    else:
        scale = max_norm / global_norm

    # 3. 保持输入列表的原始层次结构输出
    result = []
    for g in gradients:
        if isinstance(g, np.ndarray):
            result.append(g.astype(float) * scale)
        elif isinstance(g, list):
            result.append([round(float(x) * scale, 4) for x in g])
        else:
            result.append(round(float(g) * scale, 4))
            
    return result