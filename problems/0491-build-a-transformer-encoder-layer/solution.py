import numpy as np

def transformer_encoder_layer(
    X: np.ndarray, 
    weights: dict, 
    num_heads: int, 
    eps: float = 1e-5
) -> np.ndarray:
    """
    仅使用 NumPy 实现标准 Transformer 编码器层前向传播。
    """
    X = np.asarray(X, dtype=float)
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads

    # ==========================================
    # 1. 多头自注意力 (Multi-Head Self-Attention)
    # ==========================================
    # 1.1 线性投影 Q, K, V: (B, T, d_model)
    Q = X @ weights['W_q']
    K = X @ weights['W_k']
    V = X @ weights['W_v']

    # 1.2 拆分多头并转置: (B, num_heads, T, d_k)
    Q_h = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_h = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_h = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # 1.3 缩放点积打分: (B, num_heads, T, T)
    scores = (Q_h @ K_h.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    # 1.4 数值稳定的 Softmax (Encoder 全双向可见，无因果掩码)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 1.5 汇聚上下文并拼接多头还原维度: (B, T, d_model)
    context = (attn_weights @ V_h).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    attn_out = context @ weights['W_o']

    # ==========================================
    # 2. Add & Norm 1 (残差与第一层归一化)
    # ==========================================
    X1 = X + attn_out
    mean1 = np.mean(X1, axis=-1, keepdims=True)
    # 题干明确要求使用总体方差 (分母 N, 即 NumPy 默认 ddof=0)
    var1 = np.var(X1, axis=-1, ddof=0, keepdims=True)
    norm1 = weights['gamma1'] * ((X1 - mean1) / np.sqrt(var1 + eps)) + weights['beta1']

    # ==========================================
    # 3. 逐位置前馈网络 (FFN)
    # ==========================================
    # 第一层线性扩张 + ReLU 激活: (B, T, d_ff)
    h = np.maximum(0.0, norm1 @ weights['W1'] + weights['b1'])
    # 第二层线性投影压缩: (B, T, d_model)
    ffn_out = h @ weights['W2'] + weights['b2']

    # ==========================================
    # 4. Add & Norm 2 (残差与第二层归一化)
    # ==========================================
    X2 = norm1 + ffn_out
    mean2 = np.mean(X2, axis=-1, keepdims=True)
    var2 = np.var(X2, axis=-1, ddof=0, keepdims=True)
    norm2 = weights['gamma2'] * ((X2 - mean2) / np.sqrt(var2 + eps)) + weights['beta2']

    return norm2