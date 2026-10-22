import numpy as np


def kmeans_plus_plus_init(X, k, seed=None):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    if not 1 <= k <= n_samples:
        raise ValueError("k must be between 1 and the number of samples")

    if seed is not None:
        np.random.seed(seed)

    # 第一个质心：从样本中均匀随机选择
    first_idx = np.random.randint(n_samples)
    chosen_indices = [first_idx]
    centroids = [X[first_idx]]

    # 每个点到最近已选质心的平方距离
    closest_dist_sq = np.sum((X - X[first_idx]) ** 2, axis=1)

    while len(centroids) < k:
        total = closest_dist_sq.sum()

        if total > 0:
            # 距离越远，下一次被选中的概率越大
            probabilities = closest_dist_sq / total
            next_idx = np.random.choice(
                n_samples,
                p=probabilities,
            )
        else:
            # 所有剩余点都与已有质心重合时，距离权重全为 0；
            # 此时从尚未选过的样本索引中均匀选一个。
            remaining = np.setdiff1d(
                np.arange(n_samples),
                chosen_indices,
            )
            next_idx = np.random.choice(remaining)

        chosen_indices.append(next_idx)
        centroids.append(X[next_idx])

        # 更新：每个点只需记录它到“最近”质心的距离
        new_dist_sq = np.sum((X - X[next_idx]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return np.asarray(centroids)