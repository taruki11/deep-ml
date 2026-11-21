import numpy as np


def grouped_query_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, num_heads: int, num_kv_heads: int
) -> np.ndarray:
  if num_heads % num_kv_heads != 0:
    raise ValueError("num_heads must be evenly divisible by num_kv_heads")

  Q = np.asarray(Q, dtype=float)
  K = np.asarray(K, dtype=float)
  V = np.asarray(V, dtype=float)

  B, L, D = Q.shape
  head_dim = D // num_heads
  group_size = num_heads // num_kv_heads

  # 1. 拆分头维度，并将形状重排为 (B, num_heads, L, head_dim)
  Q_h = Q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
  K_h = K.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
  V_h = V.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

  # 2. 按连续分组复制 KV 头以匹配 Query 头数量
  if group_size > 1:
    K_h = np.repeat(K_h, group_size, axis=1)
    V_h = np.repeat(V_h, group_size, axis=1)

  # 3. 缩放点积计算注意力分数: (B, num_heads, L, L)
  scores = (Q_h @ K_h.swapaxes(-1, -2)) / np.sqrt(head_dim)

  # 4. 沿最后一维计算数值稳定的 Safe Softmax
  scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
  exp_scores = np.exp(scores_shifted)
  attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

  # 5. 加权聚合 Value 并还原为 (B, L, num_heads * head_dim)
  out = attn_weights @ V_h  # (B, num_heads, L, head_dim)
  out = out.transpose(0, 2, 1, 3).reshape(B, L, D)

  return out