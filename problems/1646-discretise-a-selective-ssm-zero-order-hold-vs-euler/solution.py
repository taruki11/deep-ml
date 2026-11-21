import numpy as np


def discretize(delta, A, B_in, method="zoh"):
  # 以只读方式转为浮点视图，绝不修改原输入数组
  delta = np.asarray(delta, dtype=float)
  A = np.asarray(A, dtype=float)
  B_in = np.asarray(B_in, dtype=float)

  # 1. 插入单维度轴，将各张量显式对齐至 (B, L, ED, N)
  delta_exp = delta[..., None]  # (B, L, ED, 1)
  A_exp = A[None, None, :, :]  # (1, 1, ED, N)
  B_exp = B_in[:, :, None, :]  # (B, L, 1, N)

  # 2. 计算公共状态转移矩阵 A_bar (两种离散化方法均使用精确指数)
  delta_A = delta_exp * A_exp  # (B, L, ED, N)
  A_bar = np.exp(delta_A)

  # 3. 根据所选离散化方法计算 B_bar
  if method == "zoh":
    zero_mask = A_exp == 0.0
    # 预先将分母中的 0 替换为 1.0，防止 np.where 求值时产生除零警告
    safe_A = np.where(zero_mask, 1.0, A_exp)
    # 使用 expm1 高精度计算 exp(delta * A) - 1；在 A == 0 处直接取极限值 delta
    coef = np.where(zero_mask, delta_exp, np.expm1(delta_A) / safe_A)
    B_bar = coef * B_exp
  elif method == "euler":
    B_bar = delta_exp * B_exp
  else:
    raise ValueError(f"Unsupported discretization method: {method}")

  return A_bar, B_bar