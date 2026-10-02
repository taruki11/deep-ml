import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = weights @ V
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    seq_len, d_model = Q.shape
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除")
    q_heads = np.split(Q, n_heads, axis=-1)
    k_heads = np.split(K, n_heads, axis=-1)
    v_heads = np.split(V, n_heads, axis=-1)
    head_outputs = [
        self_attention(q_i, k_i, v_i) 
        for q_i, k_i, v_i in zip(q_heads, k_heads, v_heads)
    ]
    output = np.concatenate(head_outputs, axis=-1)
    return output