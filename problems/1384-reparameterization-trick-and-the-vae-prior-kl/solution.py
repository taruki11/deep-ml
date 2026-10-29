import numpy as np


def reparameterize(mu, logvar, eps):
    """重参数化采样: z = mu + sigma * eps

    输入:
        mu: 均值矩阵，形状 (batch, latent_dim)
        logvar: 对数方差矩阵 ln(sigma^2)，形状 (batch, latent_dim)
        eps: 外部注入的标准正态分布随机噪声，形状 (batch, latent_dim)
    输出:
        z: 潜变量采样矩阵，形状 (batch, latent_dim)
    """
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    eps = np.asarray(eps, dtype=float)

    # 1. 由 log(sigma^2) 计算标准差 sigma = exp(0.5 * logvar)
    std = np.exp(0.5 * logvar)

    # 2. 逐元素仿射变换 (Hadamard 积)
    z = mu + std * eps
    return z


def kl_to_standard_normal(mu, logvar) -> float:
    """计算 q(z|x) 到 N(0, I) 的 KL 散度

    对潜变量维度求和，并在 batch 维度求平均
    输入:
        mu: 形状 (batch, latent_dim)
        logvar: 形状 (batch, latent_dim)
    输出:
        float 标量值
    """
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)

    batch_size = mu.shape[0]

    # KL = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
    kl_matrix = 0.5 * (mu**2 + np.exp(logvar) - 1.0 - logvar)

    # 在潜空间求和，在 Batch 上求均值
    total_kl = np.sum(kl_matrix) / batch_size
    return float(total_kl)