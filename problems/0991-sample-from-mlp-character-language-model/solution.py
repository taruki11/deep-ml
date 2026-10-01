import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size: int, itos: dict, seed: int, max_tokens: int) -> str:
    # 转换为 NumPy 数组
    C = np.asarray(C)
    W1 = np.asarray(W1)
    b1 = np.asarray(b1)
    W2 = np.asarray(W2)
    b2 = np.asarray(b2)
    vocab_size = C.shape[0]
    
    # 1. 严格在函数最开始创建全局唯一的随机数生成器
    rng = np.random.default_rng(seed)
    
    # 2. 初始上下文窗口全为 0 (代表 '.')
    context = [0] * block_size
    out_chars = []
    
    # 3. 自回归循环生成
    for _ in range(max_tokens):
        # 步骤 A: 查表与展平
        # C[context] 形状: (block_size, emb_dim) -> 展平成一维 (block_size * emb_dim,)
        emb_flat = C[context].flatten()
        
        # 步骤 B: 隐藏层前向 (带 tanh 激活)
        h = np.tanh(emb_flat @ W1 + b1)
        
        # 步骤 C: 输出层前向得到 logits
        logits = h @ W2 + b2
        
        # 步骤 D: 数值稳定版 Softmax 计算概率
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)
        
        # 步骤 E: 按概率采样下一个字符 ID
        ix = int(rng.choice(vocab_size, p=probs))
        
        # 步骤 F: 遇到结束符 0 ('.') 则停止生成
        if ix == 0:
            break
            
        # 步骤 G: 记录字符并滑动上下文窗口
        out_chars.append(itos[ix])
        context = context[1:] + [ix]  # 弹出最左侧，右侧追加新生成的字符
        
    return "".join(out_chars)