import numpy as np


def multiquery_attention(
    X: np.ndarray,
    W_queries: list,
    W_key: np.ndarray,
    W_value: np.ndarray,
    W_out: np.ndarray,
) -> np.ndarray:
  """Compute Multi-Query Attention.

  Args:
      X: Input array of shape (seq_len, d_model)
      W_queries: List of query weight matrices, each (d_model, d_k), one per
        head
      W_key: Shared key weight matrix of shape (d_model, d_k)
      W_value: Shared value weight matrix of shape (d_model, d_v)
      W_out: Output projection matrix of shape (num_heads * d_v, d_model)

  Returns:
      Output array of shape (seq_len, d_model), rounded to 4 decimal places
  """
  X = np.asarray(X, dtype=float)
  W_key = np.asarray(W_key, dtype=float)
  W_value = np.asarray(W_value, dtype=float)
  W_out = np.asarray(W_out, dtype=float)

  # 1. 所有头共享同一份 Key 和 Value 投影，全局仅计算一次
  K = X @ W_key  # (seq_len, d_k)
  V = X @ W_value  # (seq_len, d_v)

  d_k = K.shape[-1]
  scale = np.sqrt(d_k)

  # 2. 对每个头独立计算 Query 投影与缩放点积注意力
  head_outputs = []
  for W_q in W_queries:
    Q = X @ np.asarray(W_q, dtype=float)  # (seq_len, d_k)
    scores = (Q @ K.T) / scale  # (seq_len, seq_len)

    # 沿最后一维执行 Safe Softmax
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    head_out = attn_weights @ V  # (seq_len, d_v)
    head_outputs.append(head_out)

  # 3. 拼接全部头的输出并通过 W_out 映射回 d_model
  concat_out = np.concatenate(head_outputs, axis=-1)  # (seq_len, num_heads * d_v)
  output = concat_out @ W_out  # (seq_len, d_model)

  return np.round(output, 4)