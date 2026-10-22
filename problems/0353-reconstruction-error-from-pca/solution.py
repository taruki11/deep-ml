import numpy as np


def pca_reconstruction_error(X, n_components):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix")

    n_samples, n_features = X.shape
    if not 0 <= n_components <= n_features:
        raise ValueError("n_components must be between 0 and n_features")

    # 按列中心化，保存均值用于还原
    mean = X.mean(axis=0)
    X_centered = X - mean

    # 题目指定 population covariance：除以 n
    covariance = X_centered.T @ X_centered / n_samples

    # eigh 用于对称矩阵；特征向量按列存放
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # 取特征值最大的前 n_components 个方向
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:n_components]]

    # 投影到低维，再投影回原空间
    Z = X_centered @ components
    X_reconstructed = Z @ components.T + mean

    # 对所有数据元素求平均平方误差
    return float(np.mean((X - X_reconstructed) ** 2))