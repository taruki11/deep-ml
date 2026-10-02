import numpy as np

def attention(Q, K, V):
    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(dim)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = weights @ V
    return output