import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    running_seq = list(idx)
    
    for _ in range(max_new_tokens):
        # 1. 滑动窗口截断：只取最后最多 context_size 个 Token
        cropped_seq = running_seq[-context_size:]
        
        # 2. 升维包装为 2D 数组: (1, T_current)
        input_tensor = np.array([cropped_seq], dtype=np.int64)
        
        # 3. 模型前向推理: (1, T_current, vocab_size)
        logits = model(input_tensor)
        
        # 4. 提取最后一个时间步在词表上的打分向量: (vocab_size,)
        # logits[0, -1, :] 表示 batch 0、最后一个 token 位置的所有词表打分
        last_token_logits = logits[0, -1, :]
        
        # 5. 贪心决策：挑选打分最高的 Token ID
        next_token = int(np.argmax(last_token_logits))
        
        # 6. 追加到序列末尾，作为下一步自回归的历史上下文
        running_seq.append(next_token)
        
    return running_seq