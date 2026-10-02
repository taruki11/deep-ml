import numpy as np

def learned_positional_encoding(token_embeddings: np.ndarray, position_embedding_table: np.ndarray, start_pos: int = 0) -> np.ndarray:
    batch_size, seq_len, d_model = token_embeddings.shape
    
    # 1. 从位置表中切取当前所需的连续位置向量: [start_pos, start_pos + seq_len)
    # pos_embeddings 形状为: (seq_len, d_model)
    pos_embeddings = position_embedding_table[start_pos : start_pos + seq_len]
    
    # 2. 逐元素相加
    # (batch_size, seq_len, d_model) + (seq_len, d_model)
    # 利用 NumPy 广播机制，pos_embeddings 自动广播到每一个 batch 样本上
    return token_embeddings + pos_embeddings