import numpy as np


def k_means_clustering(points, k, initial_centroids, max_iterations):
    X = np.asarray(points, dtype=float)                  # (n, d)
    centroids = np.asarray(initial_centroids, dtype=float)  # (k, d)

    for _ in range(max_iterations):
        # 距离矩阵形状为 (n, k)：每行一个点，每列一个质心
        distances = np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2,
        )

        # 每个点分给最近的质心；距离并列时选索引靠前的质心
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()

        for j in range(k):
            cluster_points = X[labels == j]

            # 如果某组当前没有分到点，保留它原来的质心
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)

        # 质心不再变化时提前结束
        if np.allclose(new_centroids, centroids, rtol=0.0, atol=1e-12):
            centroids = new_centroids
            break

        centroids = new_centroids

    # 最后才四舍五入，避免舍入误差影响迭代
    return [
        tuple(round(float(value), 4) for value in center)
        for center in centroids
    ]