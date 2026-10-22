import numpy as np


def bagging_classifier(X_train, y_train, X_test, n_estimators=10, seed=42):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)

    n_samples, n_features = X_train.shape

    if n_samples == 0:
        raise ValueError("X_train must not be empty")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")

    # 经典 NumPy 随机数生成器；固定 seed 后抽样可复现
    rng = np.random.RandomState(seed)

    def fit_stump(X, y):
        # 纯节点无法从类别标签中学到切分，直接预测唯一类别
        if np.all(y == y[0]):
            return None, None, None, int(y[0])

        best_error = len(y) + 1
        best_stump = None

        for feature in range(n_features):
            values = np.unique(X[:, feature])

            # 相邻不同取值的中点作为候选阈值
            thresholds = [
                float(values[i] + (values[i + 1] - values[i]) / 2.0)
                for i in range(len(values) - 1)
            ]

            for threshold in thresholds:
                left_mask = X[:, feature] < threshold

                # direction 0：左边预测 0，右边预测 1
                # direction 1：左边预测 1，右边预测 0
                for left_label in (0, 1):
                    predictions = np.where(
                        left_mask,
                        left_label,
                        1 - left_label
                    )
                    error = int(np.sum(predictions != y))

                    # 只在错误数严格更小时更新，平局保留先扫描到的方案
                    if error < best_error:
                        best_error = error
                        best_stump = (feature, threshold, left_label, None)

        # 所有特征都没有可切分阈值时，退化为预测多数类
        if best_stump is None:
            majority = int(np.bincount(y, minlength=2).argmax())
            return None, None, None, majority

        return best_stump

    all_predictions = []

    for _ in range(n_estimators):
        # 有放回地抽 n_samples 次，组成一个 bootstrap 样本
        sample_indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )
        X_bootstrap = X_train[sample_indices]
        y_bootstrap = y_train[sample_indices]

        stump = fit_stump(X_bootstrap, y_bootstrap)
        feature, threshold, left_label, constant_label = stump

        if feature is None:
            predictions = np.full(
                len(X_test),
                constant_label,
                dtype=int
            )
        else:
            left_mask = X_test[:, feature] < threshold
            predictions = np.where(
                left_mask,
                left_label,
                1 - left_label
            )

        all_predictions.append(predictions)

    # 形状为 (n_estimators, 测试样本数)
    votes = np.asarray(all_predictions)

    # 1 类必须获得严格多数票；平票时返回 0
    return (votes.sum(axis=0) > n_estimators / 2).astype(int)