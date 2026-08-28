import math


def calculate_eigenvalues(
    matrix: list[list[float | int]],
) -> list[float]:
  a, b = matrix[0][0], matrix[0][1]
  c, d = matrix[1][0], matrix[1][1]

  # 1. 计算迹 (trace) 和 行列式 (det)
  trace = a + d
  det = a * d - b * c

  # 2. 判别式
  delta = trace**2 - 4 * det

  # 3. 求两根（因为 + 开根号的一定更大，所以直接就是降序）
  sqrt_delta = math.sqrt(delta)
  lambda1 = (trace + sqrt_delta) / 2
  lambda2 = (trace - sqrt_delta) / 2

  return [round(float(lambda1), 4), round(float(lambda2), 4)]