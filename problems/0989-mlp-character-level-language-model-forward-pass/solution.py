import numpy as np

def mlp_forward_loss(X, Y, C, W1, b1, W2, b2) -> float:
    # 转换为 NumPy 数组
    X = np.asarray(X)
    Y = np.asarray(Y)
    C = np.asarray(C)
    W1 = np.asarray(W1)
    b1 = np.asarray(b1)
    W2 = np.asarray(W2)
    b2 = np.asarray(b2)
    
    N = X.shape[0]
    
    # 步骤 1: 嵌入查表
    # X 形状 (N, block_size) -> emb 形状 (N, block_size, emb_dim)
    emb = C[X]
    
    # 步骤 2: 展平拼接上下文向量
    # emb.reshape(N, -1) 自动推导出后两维展平为 (N, block_size * emb_dim)
    emb_flat = emb.reshape(N, -1)
    
    # 步骤 3: 计算全连接隐藏层 (带 tanh 非线性激活)
    # (N, block_size * emb_dim) @ (block_size * emb_dim, hidden) + (hidden,) -> (N, hidden)
    h = np.tanh(emb_flat @ W1 + b1)
    
    # 步骤 4: 计算输出层 Logits
    # (N, hidden) @ (hidden, vocab_size) + (vocab_size,) -> (N, vocab_size)
    logits = h @ W2 + b2
    
    # 步骤 5: 数值稳定版 Softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 步骤 6: 提取正确标签位置的概率，计算平均负对数似然 (交叉熵)
    correct_probs = probs[np.arange(N), Y]
    loss = -np.mean(np.log(correct_probs))
    
    return float(loss)