# --- cell 1 ---
import numpy as np


class MyReducer:
    def __init__(self):
        self.n_components = 10
        self.mean_ = None
        self.projection_ = None

    def fit(self, X):
        """从训练数据中学习均值和10个主成分方向。"""
        X = np.asarray(X, dtype=np.float64)

        # 每个像素列的训练集均值，形状 (n_features,)
        self.mean_ = X.mean(axis=0)

        # 中心化：每个像素减去训练集均值，形状 (n_samples, n_features)
        X_centered = X - self.mean_

        # X_centered = U @ diag(S) @ Vt
        # Vt 的前10行是解释方差最大的10个方向
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 投影矩阵形状必须是 (n_features, 10)
        self.projection_ = np.zeros(
            (X.shape[1], self.n_components),
            dtype=np.float64
        )
        n_available = min(self.n_components, Vt.shape[0])
        self.projection_[:, :n_available] = Vt[:n_available].T

        # 奇异向量的正负号数学上都有效；统一方向，保证输出稳定
        for j in range(n_available):
            largest = np.argmax(np.abs(self.projection_[:, j]))
            if self.projection_[largest, j] < 0:
                self.projection_[:, j] *= -1

        return self

    def transform(self, X):
        """用 fit 学到的均值和投影矩阵转换新数据。"""
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.projection_

    def fit_transform(self, X):
        """先学习降维方向，再转换这批数据。"""
        return self.fit(X).transform(X)