import numpy as np

def top_p_sampling(logits: list, p: float) -> list:
    logits = np.asarray(logits, dtype=float)
    
    # 1. 向量化 Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # 2. 稳定排序：降序排列索引（kind='stable' 天然满足平局时小索引在前的规则）
    sort_idx = np.argsort(-probs, kind='stable')
    sorted_probs = probs[sort_idx]
    
    # 3. 核心向量化技巧：错位累积和 (Shifted Cumsum) 判定阈值
    # (cum_probs - sorted_probs) 正好表示“前一个词的累积概率”，必须小于 p 才能保留当前词
    cum_probs = np.cumsum(sorted_probs)
    keep_mask = (cum_probs - sorted_probs) < p
    
    # 4. 掩码过滤并就地重归一化
    sorted_probs[~keep_mask] = 0.0
    sorted_probs /= np.sum(sorted_probs)
    
    # 5. 一行代码将排序后的概率写回原始位置（逆向散射）
    out = np.zeros_like(probs)
    out[sort_idx] = sorted_probs
    
    return np.round(out, 4).tolist()