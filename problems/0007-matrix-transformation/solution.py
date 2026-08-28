import numpy as np



def transform_matrix(A, T, S):
  try:
    # 1. 顺便算一下 T 和 S 的逆矩阵（如果不可逆，直接崩溃被 except 捕获）
    T_inv = np.linalg.inv(T)
    _ = np.linalg.inv(S)

    # 2. 矩阵乘法（如果维度不匹配，直接崩溃被 except 捕获）
    return (T_inv @ np.array(A) @ np.array(S)).tolist()
  except Exception:
    # 任何维度不匹配、矩阵不可逆、非方阵，统统进这里返回 -1
    return -1