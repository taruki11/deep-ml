import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A = np.asarray(A, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("A must have shape (2, 2)")

    # 右奇异向量来自 A.T @ A
    B = A.T @ A
    a = B[0, 0]
    b = B[0, 1]
    d = B[1, 1]

    # 一次 Jacobi 旋转，对角化 B
    theta = 0.5 * np.arctan2(2.0 * b, a - d)
    c = np.cos(theta)
    s = np.sin(theta)

    V = np.array([
        [c, -s],
        [s,  c],
    ])

    # 旋转 A 的列；旋转后两列互相垂直
    AV = A @ V

    # 两列长度就是奇异值
    S = np.linalg.norm(AV, axis=0)

    # 按奇异值从大到小排列
    order = np.argsort(-S)
    S = S[order]
    V = V[:, order]
    AV = AV[:, order]

    # 将 AV 的列单位化，得到 U
    U = np.zeros((2, 2), dtype=float)
    tolerance = (
        np.finfo(float).eps
        * max(float(S[0]), np.finfo(float).tiny)
        * 10
    )

    if S[0] > tolerance:
        U[:, 0] = AV[:, 0] / S[0]

        if S[1] > tolerance:
            U[:, 1] = AV[:, 1] / S[1]
        else:
            # 第二个奇异值为 0 时，补一个与第一列垂直的单位向量
            U[:, 1] = [-U[1, 0], U[0, 0]]
    else:
        # A 是零矩阵时，U 可以取任意正交矩阵
        U = np.eye(2)

    Vt = V.T
    return U, S, Vt