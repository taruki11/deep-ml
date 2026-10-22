import numpy as np


def dbscan(X, eps, min_samples):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    # 用平方距离比较，等价于欧氏距离 <= eps，
    # 但省去了开平方。
    diff = X[:, None, :] - X[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    neighbor_mask = dist_sq <= eps ** 2

    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)

    cluster_id = 0

    for i in range(n_samples):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = np.flatnonzero(neighbor_mask[i]).tolist()

        # 邻居数包含点自身
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        # 找到核心点，创建一个新簇并从它开始扩展
        labels[i] = cluster_id
        queue = list(neighbors)
        queued = set(queue)
        pos = 0

        while pos < len(queue):
            point = queue[pos]
            pos += 1

            if not visited[point]:
                visited[point] = True
                point_neighbors = np.flatnonzero(
                    neighbor_mask[point]
                ).tolist()

                # point 是核心点，就把它的邻居也加入扩展队列
                if len(point_neighbors) >= min_samples:
                    for neighbor in point_neighbors:
                        if neighbor not in queued:
                            queue.append(neighbor)
                            queued.add(neighbor)

            # 之前标成噪声的点可能是某个簇的边界点，
            # 此时可以把它收进当前簇。已属于其他簇的点不改标签。
            if labels[point] == -1:
                labels[point] = cluster_id

        cluster_id += 1

    return labels