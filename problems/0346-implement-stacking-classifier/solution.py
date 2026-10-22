import numpy as np


def stacking_classifier(
    X_train,
    y_train,
    X_test,
    base_classifiers,
    meta_classifier,
    n_folds=5
):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    n_samples = len(y_train)
    n_base = len(base_classifiers)

    if n_folds < 2 or n_folds > n_samples:
        raise ValueError("n_folds must be between 2 and n_samples")

    # meta_train 的一行对应一个训练样本；每一列对应一个基础分类器
    meta_train = np.empty((n_samples, n_base), dtype=float)

    fold_size = n_samples // n_folds

    # 创建训练集折外预测
    for fold in range(n_folds):
        start = fold * fold_size

        # 前 n_folds - 1 折大小为 fold_size；
        # 最后一折接收剩余所有样本
        end = start + fold_size if fold < n_folds - 1 else n_samples

        validation_indices = np.arange(start, end)
        fitting_indices = np.concatenate([
            np.arange(0, start),
            np.arange(end, n_samples)
        ])

        X_fit = X_train[fitting_indices]
        y_fit = y_train[fitting_indices]
        X_held_out = X_train[validation_indices]

        # 每个基础分类器分别预测当前未参与训练的折
        for j, clf in enumerate(base_classifiers):
            predictions = np.asarray(
                clf(X_fit, y_fit, X_held_out)
            ).reshape(-1)

            if len(predictions) != len(validation_indices):
                raise ValueError(
                    "Each base classifier must return one prediction per row"
                )

            meta_train[validation_indices, j] = predictions

    # 测试集的元特征：基础分类器各自用完整训练集训练
    meta_test = np.empty((len(X_test), n_base), dtype=float)

    for j, clf in enumerate(base_classifiers):
        predictions = np.asarray(
            clf(X_train, y_train, X_test)
        ).reshape(-1)

        meta_test[:, j] = predictions

    # 元分类器用 OOF 预测训练，再预测测试集
    final_predictions = meta_classifier(meta_train, y_train, meta_test)

    return np.asarray(final_predictions, dtype=int).reshape(-1)