import numpy as np


def dense_backward(
    a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray
) -> dict:
  """计算全连接层平方误差损失关于 W, b, a_prev 的梯度。

  Args:
      a_prev: 上一层激活值，形状为 (K,)
      W: 权重矩阵，形状为 (J, K)
      b: 偏置向量，形状为 (J,)
      y: 真实标签向量，形状为 (J,)

  Returns:
      字典，包含 'dW' (J x K 嵌套列表), 'db' (长为 J 的列表), 'da_prev' (长为 K 的列表)
  """
  a_prev = np.asarray(a_prev, dtype=float)
  W = np.asarray(W, dtype=float)
  b = np.asarray(b, dtype=float)
  y = np.asarray(y, dtype=float)

  # 1. 前向传播
  z = W @ a_prev + b
  a = 1.0 / (1.0 + np.exp(-z))

  # 2. 误差项 delta = dC/dz = 2 * (a - y) * a * (1 - a)
  delta = 2.0 * (a - y) * a * (1.0 - a)

  # 3. 梯度反传
  dW = np.outer(delta, a_prev)  # (J, K)
  db = delta  # (J,)
  da_prev = W.T @ delta  # (K,) 聚合所有 J 个下游神经元的梯度贡献

  # 题目严格要求：不进行内部四舍五入，直接返回原生 Python 列表
  return {"dW": dW.tolist(), "db": db.tolist(), "da_prev": da_prev.tolist()}