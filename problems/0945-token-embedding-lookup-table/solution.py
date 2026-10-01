import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list[int], seed: int) -> list[list[float]]:
    # 严格按照题目要求使用 default_rng 和 standard_normal
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal(size=(vocab_size, embed_dim))
    
    # 查表并以嵌套列表形式返回
    return weights[token_ids].tolist()