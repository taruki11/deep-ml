import numpy as np


def poly_fit(x, r, degree):
    """
    用 1, x, x^2, ..., x^degree 拟合 r。
    返回系数 [常数项, x 的系数, x^2 的系数, ...]。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)

    # 每一行对应一个样本，每一列依次是 1, x, x^2, ...
    design = np.vander(x, degree + 1, increasing=True)

    coef = np.linalg.lstsq(design, r, rcond=None)[0]
    return coef


def poly_predict(x, coef):
    """用多项式系数计算预测值。"""
    x = np.asarray(x, dtype=float)
    coef = np.asarray(coef, dtype=float)

    result = np.zeros_like(x, dtype=float)
    for power, weight in enumerate(coef):
        result += weight * (x ** power)

    return result


def backfit_gam(X, y, degree, n_iter=20):
    """
    用 backfitting 拟合加性模型。

    X: (n, p)，n 个样本、p 个特征
    y: (n,)
    返回：
      alpha: 截距
      coefs: p 个多项式系数向量
      F: (n, p)，每个特征对每个样本的拟合贡献
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n, p = X.shape

    # 初始截距取 y 的均值；每个特征的初始贡献为 0
    alpha = float(np.mean(y))
    F = np.zeros((n, p), dtype=float)
    coefs = [np.zeros(degree + 1, dtype=float) for _ in range(p)]

    # 每一轮按顺序更新每个特征的函数
    for _ in range(n_iter):
        for j in range(p):
            # 当前第 j 列旧贡献先排除，只减去其他列的贡献
            other_contributions = F.sum(axis=1) - F[:, j]
            partial_residual = y - alpha - other_contributions

            # 用第 j 个特征拟合部分残差
            coef = poly_fit(X[:, j], partial_residual, degree)
            fitted = poly_predict(X[:, j], coef)

            # 把这个函数中心化：让训练集上的平均贡献为 0
            fitted_mean = np.mean(fitted)
            fitted = fitted - fitted_mean
            coef[0] -= fitted_mean

            # 保存更新后的函数
            F[:, j] = fitted
            coefs[j] = coef

    return alpha, coefs, F


def gam_predict(X_new, alpha, coefs):
    """用训练好的截距和各个多项式函数预测新数据。"""
    X_new = np.asarray(X_new, dtype=float)

    predictions = np.full(X_new.shape[0], float(alpha), dtype=float)

    for j, coef in enumerate(coefs):
        predictions += poly_predict(X_new[:, j], coef)

    return predictions