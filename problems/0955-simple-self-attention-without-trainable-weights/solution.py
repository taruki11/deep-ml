import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    X = np.asarray(X, dtype=float)
    scores = X @ X.T
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    Z = attn_weights @ X
    return Z.tolist()
    pass
