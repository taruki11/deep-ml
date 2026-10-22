import numpy as np


def gradient_boosting_step(
    X,
    y,
    current_predictions,
    learning_rate=0.1
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    current_predictions = np.asarray(current_predictions, dtype=float)

    n_samples, n_features = X.shape
    residuals = y - current_predictions

    best_mse = float("inf")
    best_split = None

    # 特征按列顺序扫描
    for feature in range(n_features):
        values = np.unique(X[:, feature])

        # 每两个相邻的不同取值之间取中点作为阈值
        for i in range(len(values) - 1):
            threshold = float(
                values[i] + (values[i + 1] - values[i]) / 2.0
            )

            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            # 每个叶子预测该叶子中残差的平均值
            left_mean = residuals[left_mask].mean()
            right_mean = residuals[right_mask].mean()

            stump_predictions = np.empty(n_samples, dtype=float)
            stump_predictions[left_mask] = left_mean
            stump_predictions[right_mask] = right_mean

            mse = float(np.mean((residuals - stump_predictions) ** 2))

            # 只在 MSE 严格更小时更新；平局时保留先扫描到的切分
            if mse < best_mse:
                best_mse = mse
                best_split = (feature, threshold)

    if best_split is None:
        # 所有特征都没有有效阈值
        stump_predictions = np.full(
            n_samples,
            residuals.mean(),
            dtype=float
        )
    else:
        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        stump_predictions = np.empty(n_samples, dtype=float)
        stump_predictions[left_mask] = residuals[left_mask].mean()
        stump_predictions[right_mask] = residuals[right_mask].mean()

    updated_predictions = (
        current_predictions + learning_rate * stump_predictions
    )

    # 只在最终返回时四舍五入
    return np.round(updated_predictions, 4).tolist()