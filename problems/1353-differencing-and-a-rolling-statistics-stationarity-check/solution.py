import numpy as np


def difference(x, d):
    x = np.asarray(x, dtype=float)

    if d == 0:
        return x.copy()

    return np.diff(x, n=d)


def stationarity_report(x, n_chunks):
    x = np.asarray(x, dtype=float)

    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")

    chunk_size = len(x) // n_chunks
    if chunk_size == 0:
        raise ValueError("每个 chunk 至少需要包含一个样本")

    # 只取能组成 n_chunks 个等长块的部分；末尾余数丢弃
    usable_length = chunk_size * n_chunks
    x = x[:usable_length]

    chunks = [
        x[i * chunk_size:(i + 1) * chunk_size]
        for i in range(n_chunks)
    ]

    chunk_means = np.array([chunk.mean() for chunk in chunks])
    chunk_stds = np.array([chunk.std(ddof=0) for chunk in chunks])

    mean_spread = float(chunk_means.max() - chunk_means.min())

    min_std = float(chunk_stds.min())
    max_std = float(chunk_stds.max())

    if min_std == 0:
        if max_std == 0:
            std_ratio = 1.0
        else:
            std_ratio = float("inf")
    else:
        std_ratio = max_std / min_std

    return round(mean_spread, 4), round(std_ratio, 4)