import numpy as np


def silhouette_score(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n = len(X)
    clusters = np.unique(labels)

    # 只有一个簇，或每个点各自一个簇
    if n <= 1 or len(clusters) <= 1 or len(clusters) == n:
        return 0.0

    # dist[i, j] 是样本 i 和样本 j 之间的欧氏距离
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    scores = np.zeros(n, dtype=float)

    for i in range(n):
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # 计算 a 时排除样本自身

        # 如果它是单点簇，轮廓值按约定设为 0
        if not np.any(same_cluster):
            continue

        a = dist[i, same_cluster].mean()

        other_means = []
        for cluster in clusters:
            if cluster != labels[i]:
                in_cluster = labels == cluster
                other_means.append(dist[i, in_cluster].mean())

        b = min(other_means)
        denominator = max(a, b)

        if denominator > 0:
            scores[i] = (b - a) / denominator

    return round(float(scores.mean()), 4)