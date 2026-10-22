import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    # 从常数项 1 开始
    columns = [np.ones(n_samples, dtype=float)]

    # 按总次数从 1 到 degree 生成项
    for total_degree in range(1, degree + 1):
        for combo in combinations_with_replacement(
            range(n_features), total_degree
        ):
            # 例如 combo=(0, 1) 表示 x0 * x1
            column = np.prod(X[:, list(combo)], axis=1)
            columns.append(column)

    # 形状：(样本数, 多项式项数量)
    features = np.column_stack(columns)

    # 每个样本内部排序
    return np.sort(features, axis=1)