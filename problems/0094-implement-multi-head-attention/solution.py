import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = K.shape[-1]

    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    seq_len, d_model = Q.shape
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除")
    
    q_heads = torch.chunk(Q, n_heads, dim=-1)
    k_heads = torch.chunk(K, n_heads, dim=-1)
    v_heads = torch.chunk(V, n_heads, dim=-1)
    
    head_outputs = [
        self_attention(q_i, k_i, v_i) 
        for q_i, k_i, v_i in zip(q_heads, k_heads, v_heads)
    ]
    
    output = torch.cat(head_outputs, dim=-1)
    return output