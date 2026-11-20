import numpy as np


def tie_weights(token_embedding, lm_head_weight):
  """Return the LM head weight matrix, tying to the token embedding if lm_head_weight is None.

  Args:
      token_embedding: np.ndarray of shape (vocab_size, hidden_dim)
      lm_head_weight: np.ndarray of shape (vocab_size, hidden_dim) or None

  Returns:
      np.ndarray to use as the LM head weight matrix
  """
  # 1. 若未提供独立输出头权重，直接返回原数组引用（共享同一块内存）
  if lm_head_weight is None:
    return token_embedding

  # 2. 若提供了独立权重，严格校验形状是否一致
  if lm_head_weight.shape != token_embedding.shape:
    raise ValueError(
        f"Shape mismatch: expected {token_embedding.shape}, got"
        f" {lm_head_weight.shape}"
    )

  # 3. 形状一致，原样返回独立权重
  return lm_head_weight
