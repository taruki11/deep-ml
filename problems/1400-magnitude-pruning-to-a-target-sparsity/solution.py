import numpy as np


def magnitude_prune(W, sparsity):
    """单矩阵非结构化幅值剪枝。
    输入:
        W: 原始权重数组 (任意维度的 numpy array 或可转为 array 的列表)
        sparsity: 目标稀疏度比例 (float, 0.0 ~ 1.0)
    输出:
        (pruned, mask):
            pruned: 与 W 同形状的 float 数组，剪除的元素为 0.0
            mask: 与 W 同形状的 bool 数组 (True 为保留，False 为剪除)
    """
    W = np.asarray(W, dtype=float)
    flat_W = W.ravel()
    n_total = flat_W.size

    # 1. 计算需剪除的参数个数
    n_prune = int(np.floor(sparsity * n_total))

    # 2. 稳定排序获得绝对值最小的前 n_prune 个平铺索引
    # kind='stable' 保证绝对值相同时，靠前的索引排在前面被剪除
    sort_indices = np.argsort(np.abs(flat_W), kind='stable')
    prune_indices = sort_indices[:n_prune]

    # 3. 构建布尔掩码与剪枝后张量
    mask_flat = np.ones(n_total, dtype=bool)
    mask_flat[prune_indices] = False

    mask = mask_flat.reshape(W.shape)
    pruned = W * mask
    return pruned, mask


def global_magnitude_prune(weights, sparsity):
    """跨多个权重的全局非结构化幅值剪枝。
    输入:
        weights: 权重矩阵构成的 list
        sparsity: 全局目标稀疏度 (float, 0.0 ~ 1.0)
    输出:
        [(pruned_1, mask_1), (pruned_2, mask_2), ...]
    """
    weights = [np.asarray(w, dtype=float) for w in weights]
    shapes = [w.shape for w in weights]
    sizes = [w.size for w in weights]
    total_size = sum(sizes)

    # 1. 计算所有层加和后需剪除的全局元素总数
    n_prune = int(np.floor(sparsity * total_size))

    # 2. 将所有权重平铺拼接为一个全局 1D 数组
    flat_all = np.concatenate([w.ravel() for w in weights])

    # 3. 全局统一稳定排序
    sort_indices = np.argsort(np.abs(flat_all), kind='stable')
    prune_indices = sort_indices[:n_prune]

    # 4. 构建全局掩码并按原始层尺寸切分还原
    mask_all = np.ones(total_size, dtype=bool)
    mask_all[prune_indices] = False

    results = []
    current_idx = 0
    for w, shape, size in zip(weights, shapes, sizes):
        sub_mask = mask_all[current_idx : current_idx + size].reshape(shape)
        sub_pruned = w * sub_mask
        results.append((sub_pruned, sub_mask))
        current_idx += size

    return results


def actual_sparsity(mask):
    """计算掩码中被置零 (False) 元素的比例。"""
    mask = np.asarray(mask, dtype=bool)
    return float(np.mean(~mask))