import numpy as np


def gmm_e_step(X, means, variances, mixing_coeffs):
    X = np.asarray(X, dtype=float).reshape(-1)
    means = np.asarray(means, dtype=float).reshape(-1)
    variances = np.asarray(variances, dtype=float).reshape(-1)
    mixing_coeffs = np.asarray(mixing_coeffs, dtype=float).reshape(-1)

    # 广播后形状为 (n_samples, n_components)
    x = X[:, None]
    mu = means[None, :]
    var = variances[None, :]

    # log[π_k * N(x_i | μ_k, σ²_k)]
    log_joint = (
        np.log(mixing_coeffs)[None, :]
        - 0.5 * (
            np.log(2.0 * np.pi * var)
            + (x - mu) ** 2 / var
        )
    )

    # 按行做稳定的 softmax 归一化
    row_max = np.max(log_joint, axis=1, keepdims=True)
    exp_values = np.exp(log_joint - row_max)

    return exp_values / np.sum(exp_values, axis=1, keepdims=True)