import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
  # 1. 取最大值防溢出
  max_score = np.max(scores)

  # 2. 减去最大值后算 exp
  shifted_scores = scores - max_score
  exp_scores = np.exp(shifted_scores)

  # 3. 按照公式：(z_i - M) - ln(sum(exp))
  return shifted_scores - np.log(np.sum(exp_scores))