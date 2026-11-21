import numpy as np


def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int,
) -> tuple:
  X = np.asarray(X, dtype=float)
  W_dkv = np.asarray(W_dkv, dtype=float)
  W_uk = np.asarray(W_uk, dtype=float)
  W_uv = np.asarray(W_uv, dtype=float)
  W_dq = np.asarray(W_dq, dtype=float)
  W_uq = np.asarray(W_uq, dtype=float)
  W_o = np.asarray(W_o, dtype=float)

  L, d_model = X.shape
  if d_model % n_heads != 0:
    raise ValueError("d_model must be evenly divisible by n_heads")
  d_head = d_model // n_heads

  # 1. KV 低秩压缩 (该潜变量即为推理期真正存入 KV Cache 的内容)
  c_kv = X @ W_dkv  # (L, d_c_kv)

  # 2. 从潜空间重构 Keys 与 Values
  K = c_kv @ W_uk  # (L, d_model)
  V = c_kv @ W_uv  # (L, d_model)

  # 3. Query 压缩与重构
  c_q = X @ W_dq  # (L, d_c_q)
  Q = c_q @ W_uq  # (L, d_model)

  # 4. 拆分为多头形式: (n_heads, L, d_head)
  Q_h = Q.reshape(L, n_heads, d_head).transpose(1, 0, 2)
  K_h = K.reshape(L, n_heads, d_head).transpose(1, 0, 2)
  V_h = V.reshape(L, n_heads, d_head).transpose(1, 0, 2)

  # 5. 缩放点积注意力 + 数值稳定 Safe Softmax
  scores = (Q_h @ K_h.swapaxes(-1, -2)) / np.sqrt(d_head)  # (n_heads, L, L)
  scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
  exp_scores = np.exp(scores_shifted)
  attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

  # 6. 多头结果聚合、拼接并执行输出投影
  head_out = attn_weights @ V_h  # (n_heads, L, d_head)
  concat = head_out.transpose(1, 0, 2).reshape(L, d_model)  # (L, d_model)
  output = concat @ W_o  # (L, d_model)

  return (output, c_kv)