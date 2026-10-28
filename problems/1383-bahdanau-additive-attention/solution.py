import numpy as np

def additive_attention(query, keys, W_q, W_k, v):
    """
    Bahdanau 加性注意力机制
    
    参数:
        query: 解码器状态，形状 (d_q,)
        keys: 编码器状态矩阵 (同时作为 values)，形状 (n, d_k)
        W_q: 查询投影矩阵，形状 (d_a, d_q)
        W_k: 键投影矩阵，形状 (d_a, d_k)
        v: 得分投影向量，形状 (d_a,)
        
    返回:
        (weights, context):
            weights: 形状为 (n,) 的归一化概率分布，和为 1
            context: 形状为 (d_k,) 的加权上下文特征向量
    """
    query = np.asarray(query, dtype=float).reshape(-1)
    keys = np.asarray(keys, dtype=float)
    W_q = np.asarray(W_q, dtype=float)
    W_k = np.asarray(W_k, dtype=float)
    v = np.asarray(v, dtype=float).reshape(-1)
    
    # 1. 投影 Query 向量: (d_a, d_q) @ (d_q,) -> (d_a,)
    query_proj = np.dot(W_q, query)
    
    # 2. 批量投影所有 Keys: (n, d_k) @ (d_a, d_k).T -> (n, d_a)
    keys_proj = np.dot(keys, W_k.T)
    
    # 3. 利用 NumPy 广播机制相加并求 tanh，再与 v 点乘获得打分:
    # (n, d_a) + (d_a,) 广播相加 -> (n, d_a)
    # (n, d_a) @ (d_a,) -> (n,)
    scores = np.dot(np.tanh(keys_proj + query_proj), v)
    
    # 4. 数值稳定的 Softmax (减去最大值防止 exp 上溢)
    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores)
    weights = exp_scores / np.sum(exp_scores)  # 形状: (n,)
    
    # 5. 上下文向量: 对 keys 进行加权求和
    # (n,) @ (n, d_k) -> (d_k,)
    context = np.dot(weights, keys)
    
    return weights, context