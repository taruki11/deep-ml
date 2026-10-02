import numpy as np


def batched_causal_self_attention(X, W_query, W_key, W_value) -> list:
    X = np.asarray(X, dtype=float)
    W_query = np.asarray(W_query, dtype=float)
    W_key = np.asarray(W_key, dtype=float)
    W_value = np.asarray(W_value, dtype=float)
    
    B, T, _ = X.shape
    d_out = W_key.shape[-1]
    
    # 步骤 1: 批量矩阵投影生成 Q, K, V
    # (B, T, d_in) @ (d_in, d_out) -> (B, T, d_out)
    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value
    
    # 步骤 2: 批量点积并除以 sqrt(d_out)
    # K 需转置最后两个维度: (B, T, d_out) -> (B, d_out, T)
    # (B, T, d_out) @ (B, d_out, T) -> (B, T, T)
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_out)
    
    # 步骤 3: 构造并广播上三角因果掩码 (k=1 排除主对角线)
    # mask 形状为 (T, T)，NumPy 会自动沿 Batch 维度广播至 (B, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    
    # 步骤 4: 沿最后一维计算数值稳定版 Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 步骤 5: 聚合计算上下文张量
    # (B, T, T) @ (B, T, d_out) -> (B, T, d_out)
    context = attn_weights @ V
    
    return context.tolist()