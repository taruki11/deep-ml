import numpy as np


def local_outlier_factor(X, k):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # 计算所有点对之间的欧氏距离，dist[i, j] 是点 i 到点 j 的距离
    differences = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))

    # 排除点自身：将距离设为无穷大，这样排序时不会选到自己
    np.fill_diagonal(distances, np.inf)

    # 稳定排序后，取每个点最近的 k 个邻居
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]

    # 每个点到第 k 个近邻的距离
    k_distances = distances[np.arange(n), neighbors[:, -1]]

    # 对 p 的每个邻居 o，计算 max(k-distance(o), distance(p, o))
    neighbor_k_distances = k_distances[neighbors]
    neighbor_distances = distances[np.arange(n)[:, None], neighbors]
    reachability_distances = np.maximum(
        neighbor_k_distances,
        neighbor_distances
    )

    # 局部可达密度：平均可达距离的倒数
    lrd = 1.0 / np.mean(reachability_distances, axis=1)

    # LOF(p) = 邻居的平均局部密度 / p 自己的局部密度
    lof = np.mean(lrd[neighbors] / lrd[:, None], axis=1)

    # 返回普通 Python list，不在函数内四舍五入
    return lof.tolist()