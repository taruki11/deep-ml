import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # 边界特判：空矩阵直接判定不兼容
    if not a or not a[0] or not b:
        return -1
        
    try:
        # 1. 尝试直接调用 numpy 的矩阵乘法
        # 2. 用 .tolist() 将 numpy 数组转回 Python 原生列表
        return np.dot(a, b).tolist()
    except (ValueError, TypeError):
        # 维度对不上时，NumPy 会抛出 ValueError，我们捕获它并返回 -1
        return -1