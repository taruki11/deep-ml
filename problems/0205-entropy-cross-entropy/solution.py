import numpy as np


def entropy_and_cross_entropy(
    P: list[float], Q: list[float]
) -> tuple[float, float]:
  # 1. 统一转为 NumPy 数组（防止 P > 0 报类型错误）
	P = np.array(P, dtype=float)
	Q = np.array(Q, dtype=float)

	# 2. 掩码过滤掉 0 项，防止 log(0) 产生 NaN
	mask = P > 0

	# 3. 计算信息熵 H(P)
	entropy = -np.sum(P[mask] * np.log(P[mask]))

	# 4. 计算交叉熵 H(P, Q)
	eps = 1e-15
	q_safe = np.clip(Q, eps, 1.0)
	cross_entropy = -np.sum(P[mask] * np.log(q_safe[mask]))

	# 5. 返回 float 元组
	return float(entropy), float(cross_entropy)