import numpy as np


def step_basis(x, cuts):
    x = np.asarray(x, dtype=float).reshape(-1)
    cuts = np.asarray(cuts, dtype=float).reshape(-1)

    # cuts 等于某个切点时，归入切点右侧的区间
    bin_index = np.searchsorted(cuts, x, side="right")

    # 每个样本所在区间对应的位置设为 1，其余为 0
    return np.eye(len(cuts) + 1, dtype=float)[bin_index]


def cubic_spline_basis(x, knots):
    x = np.asarray(x, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)

    # 基础列：x, x^2, x^3
    basis = [x, x ** 2, x ** 3]

    # 每个结点增加一列 max(x - knot, 0)^3
    for knot in knots:
        basis.append(np.maximum(x - knot, 0.0) ** 3)

    return np.column_stack(basis)


def natural_spline_basis(x, knots):
    x = np.asarray(x, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)

    K = len(knots)
    if K < 2:
        raise ValueError("natural_spline_basis requires at least 2 knots")

    # positive[i, k] = max(x_i - knot_k, 0)^3
    positive = np.maximum(x[:, None] - knots[None, :], 0.0) ** 3

    # d_1, ..., d_(K-1)
    d = (
        positive[:, :-1] - positive[:, [-1]]
    ) / (knots[-1] - knots[:-1])[None, :]

    # d_k - d_(K-1), k=1,...,K-2
    extra = d[:, :-1] - d[:, [-1]]

    # N_1 = 1, N_2 = x，后面接自然样条列
    return np.column_stack([
        np.ones(len(x), dtype=float),
        x,
        extra
    ])


def fit_predict(B_train, y_train, B_test, intercept=True):
    B_train = np.asarray(B_train, dtype=float)
    B_test = np.asarray(B_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    if intercept:
        B_train = np.column_stack([
            np.ones(B_train.shape[0], dtype=float),
            B_train
        ])
        B_test = np.column_stack([
            np.ones(B_test.shape[0], dtype=float),
            B_test
        ])

    # 最小二乘解；lstsq 对列相关或不可逆情况比直接求逆稳妥
    coefficients = np.linalg.lstsq(B_train, y_train, rcond=None)[0]

    return B_test @ coefficients