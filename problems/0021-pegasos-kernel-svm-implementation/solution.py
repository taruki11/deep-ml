import numpy as np

def pegasos_kernel_svm(
    data,
    labels,
    kernel="linear",
    lambda_val=0.01,
    iterations=100,
    sigma=1.0
):
    X = np.asarray(data, dtype=float)
    y = np.asarray(labels, dtype=float)

    n = X.shape[0]

    if kernel == "linear":
        K = X @ X.T
    elif kernel == "rbf":
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        squared_norms = np.sum(X ** 2, axis=1)
        squared_distances = (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2 * (X @ X.T)
        )
        squared_distances = np.maximum(squared_distances, 0.0)

        K = np.exp(-squared_distances / (2 * sigma ** 2))
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    alphas = np.zeros(n, dtype=float)
    bias = 0.0

    for t in range(1, iterations + 1):
        eta = 1.0 / (lambda_val * t)

        for i in range(n):
            # 使用当前的 alphas 和 bias 计算分数
            score = np.sum(alphas * y * K[:, i]) + bias

            if y[i] * score < 1:
                alphas[i] += eta * (1.0 - lambda_val * alphas[i])
                bias += eta * y[i]
            else:
                alphas[i] -= eta * lambda_val * alphas[i]

    return alphas.tolist(), float(bias)