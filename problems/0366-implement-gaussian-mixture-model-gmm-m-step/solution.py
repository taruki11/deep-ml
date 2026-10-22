import numpy as np


def gmm_m_step(X, gamma):
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    n_samples, n_features = X.shape
    n_components = gamma.shape[1]

    # 每个高斯分布获得的加权样本数
    effective_counts = gamma.sum(axis=0)

    # 若某个分量的责任度总和为 0，安全地避免除以 0；
    # 空分量的均值、协方差会设为 0，混合权重为 0。
    safe_counts = np.where(effective_counts > 0, effective_counts, 1.0)

    # means: (K, D)
    means = (gamma.T @ X) / safe_counts[:, None]

    # deviations: (N, K, D)
    deviations = X[:, None, :] - means[None, :, :]

    # covariances: (K, D, D)
    covariances = np.einsum(
        "nk,nkd,nke->kde",
        gamma,
        deviations,
        deviations,
    )
    covariances /= safe_counts[:, None, None]

    # mixing_coeffs: (K,)
    mixing_coeffs = effective_counts / n_samples

    return means, covariances, mixing_coeffs