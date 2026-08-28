import numpy as np


def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
  # 1. 先将列表转为 ndarray
  # 2. 与 scalar 相乘（NumPy 广播机制逐元素相乘）
  # 3. tolist() 转回原生二维列表
  return (np.array(matrix) * scalar).tolist()