import numpy as np


def spectral_cluster(W, k):
    W = np.asarray(W, dtype=float)
    n = W.shape[0]

    # D 的对角线是每个节点连出去的权重总和
    degrees = W.sum(axis=1)
    inv_sqrt_degrees = 1.0 / np.sqrt(degrees)

    # L_sym = I - D^(-1/2) W D^(-1/2)
    L_sym = (
        np.eye(n)
        - inv_sqrt_degrees[:, None]
        * W
        * inv_sqrt_degrees[None, :]
    )

    # eigh 返回特征值和特征向量；特征值按升序排列
    _, eigenvectors = np.linalg.eigh(L_sym)
    U = eigenvectors[:, :k]

    # 将 U 的每一行归一化为长度 1；全零行保持为零
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    T = np.divide(
        U,
        row_norms,
        out=np.zeros_like(U),
        where=row_norms > 0,
    )

    # K-means++ 风格的确定性初始化：
    # 第一个中心取第 0 行
    selected = [0]
    centers = [T[0].copy()]

    # 后续中心取“离已选中心最近距离”最大的行；
    # 并列时 np.argmax 选索引较小者。
    while len(centers) < k:
        dist_sq = np.sum(
            (T[:, None, :] - np.asarray(centers)[None, :, :]) ** 2,
            axis=2,
        )
        nearest_dist_sq = dist_sq.min(axis=1)

        # 不重复选已经作为中心的行
        nearest_dist_sq[selected] = -np.inf
        next_index = int(np.argmax(nearest_dist_sq))

        selected.append(next_index)
        centers.append(T[next_index].copy())

    centers = np.asarray(centers)

    # 在嵌入空间 T 中运行 K-means，最多 100 次
    labels = np.full(n, -1, dtype=int)

    for _ in range(100):
        dist_sq = np.sum(
            (T[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )

        # np.argmin 在距离并列时选中心索引较小者
        new_labels = np.argmin(dist_sq, axis=1)

        # 分配结果稳定时停止
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels

        # 更新质心；空簇保留原中心
        for cluster_id in range(k):
            members = T[labels == cluster_id]
            if len(members) > 0:
                centers[cluster_id] = members.mean(axis=0)

    # 按节点首次出现的顺序重新编号：
    # 第一个出现的簇编号为 0，之后遇到的新簇依次编号。
    relabel = {}
    result = []

    for label in labels:
        label = int(label)
        if label not in relabel:
            relabel[label] = len(relabel)
        result.append(relabel[label])

    return result