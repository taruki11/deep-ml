import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

import numpy as np

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    
    # 获取 key 向量的特征维度 d_k
    d_k = K.shape[-1]
    
    # 步骤 1: 点积并除以 sqrt(d_k) 缩放
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # 步骤 2: 行级数值稳定版 Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 步骤 3: 矩阵乘法计算最终上下文输出
    return weights @ V
