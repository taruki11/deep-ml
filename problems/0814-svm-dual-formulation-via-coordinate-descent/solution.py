import numpy as np

def svm_dual_coordinate_descent(X, y, C=1.0, n_iters=1000, tol=1e-9):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape
    alphas = np.zeros(n_samples, dtype=float)
    w = np.zeros(n_features, dtype=float)

    for _ in range(n_iters):
        largest_change = 0.0

        # 按要求依次更新 alpha_0, alpha_1, ..., alpha_(n-1)
        for i in range(n_samples):
            norm_sq = float(X[i] @ X[i])

            if norm_sq == 0.0:
                # x_i 为零向量时，该坐标的最优值是上界 C
                new_alpha = float(C)
            else:
                margin = float(y[i] * (X[i] @ w))
                new_alpha = alphas[i] + (1.0 - margin) / norm_sq
                new_alpha = float(np.clip(new_alpha, 0.0, C))

            change = new_alpha - alphas[i]

            # w = sum(alpha_j * y_j * x_j)，同步更新 w
            w += change * y[i] * X[i]
            alphas[i] = new_alpha

            largest_change = max(largest_change, abs(change))

        # 一整轮中所有 alpha 的最大变化都小于 tol，就停止
        if largest_change < tol:
            break

    return alphas.tolist()