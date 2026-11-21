import numpy as np


def sliding_window_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int
) -> np.ndarray:
  Q = np.asarray(Q, dtype=float)
  K = np.asarray(K, dtype=float)
  V = np.asarray(V, dtype=float)

  seq_len, d_k = Q.shape

  # 1. 计算缩放点积原始得分: (seq_len, seq_len)
  scores = (Q @ K.T) / np.sqrt(d_k)

  # 2. 构造 |i - j| > window_size 的局部窗口掩码
  # idx[:, None] 形状 (L, 1)，idx[None, :] 形状 (1, L)
  # 广播相减后得到 (L, L) 的距离矩阵，第 (i, j) 项即为 i - j
  idx = np.arange(seq_len)
  out_of_window = np.abs(idx[:, None] - idx[None, :]) > window_size
  scores = np.where(out_of_window, -np.inf, scores)

  # 3. 沿最后一维计算数值稳定的 Safe Softmax
  # 由于主对角线 |i - i| = 0 <= window_size 恒为有限值，每行最大值必不为 -inf
  scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
  exp_scores = np.exp(scores_shifted)
  attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

  # 4. 加权聚合 Value 并四舍五入至 4 位小数
  output = attn_weights @ V
  return np.round(output, 4)