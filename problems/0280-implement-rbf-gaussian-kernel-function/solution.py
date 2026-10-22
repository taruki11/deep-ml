import numpy as np

def rbf_kernel(X1, X2, gamma):
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    # 分别计算每一行的平方长度
    norm1 = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    norm2 = np.sum(X2 ** 2, axis=1, keepdims=True).T  # (1, n2)

    # 所有 X1 行与 X2 行之间的平方距离，形状 (n1, n2)
    squared_distances = norm1 + norm2 - 2 * X1 @ X2.T
    squared_distances = np.maximum(squared_distances, 0)  # 避免浮点误差产生微小负数

    return np.exp(-gamma * squared_distances)