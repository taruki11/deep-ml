import numpy as np

def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    x = np.asarray(x, dtype=float)
    W_qkv = np.asarray(W_qkv, dtype=float)
    W_out = np.asarray(W_out, dtype=float)
    batch, num_tokens, d_in = x.shape
    d_out = W_out.shape[0]
    head_dim = d_out // num_heads
    qkv = x @ W_qkv
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(batch, num_tokens, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, num_tokens, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, num_tokens, num_heads, head_dim).transpose(0, 2, 1, 3)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    causal_mask = np.triu(np.full((num_tokens, num_tokens), -np.inf), k=1)
    scores = scores + causal_mask
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    context = weights @ v
    context = context.transpose(0, 2, 1, 3).reshape(batch, num_tokens, d_out)
    output = context @ W_out
    return output.tolist()