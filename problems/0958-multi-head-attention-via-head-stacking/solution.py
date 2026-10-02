import numpy as np

def multi_head_attention_stack(x, heads) -> np.ndarray:
    """
    通过显式独立堆叠多个单头计算多头因果自注意力。
    
    参数:
        x: 输入张量，形状为 (B, T, d_in)
        heads: 列表，长度为 H。每个元素为 (W_q, W_k, W_v) 元组，矩阵形状为 (d_in, d_head)
    返回:
        output: 形状为 (B, T, H * d_head) 的 numpy 数组
    """
    x = np.asarray(x, dtype=float)
    B, T, d_in = x.shape
    
    # 构造通用的上三角因果掩码 (严格主对角线上方全置为 -inf)
    # mask 形状: (T, T)
    causal_mask = np.triu(np.full((T, T), -np.inf), k=1)
    
    head_contexts = []
    
    for W_q, W_k, W_v in heads:
        W_q = np.asarray(W_q, dtype=float)
        W_k = np.asarray(W_k, dtype=float)
        W_v = np.asarray(W_v, dtype=float)
        
        d_head = W_q.shape[-1]
        
        # 1. 批量投影: (B, T, d_in) @ (d_in, d_head) -> (B, T, d_head)
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v
        
        # 2. 批量计算缩放点积打分: (B, T, d_head) @ (B, d_head, T) -> (B, T, T)
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
        
        # 3. 施加因果掩码 (利用 NumPy 广播机制作用在最后两维)
        scores = scores + causal_mask
        
        # 4. 数值稳定的 Softmax (按行减去最大值)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 5. 加权计算当前头的上下文输出: (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
        context = weights @ V
        head_contexts.append(context)
        
    # 6. 将所有 H 个头的输出沿最后一个维度拼接: (B, T, H * d_head)
    output = np.concatenate(head_contexts, axis=-1)
    return output