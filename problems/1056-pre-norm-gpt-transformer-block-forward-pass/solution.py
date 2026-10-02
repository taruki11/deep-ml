import numpy as np
def gelu(z: np.ndarray) -> np.ndarray:
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * ((x - mean) / np.sqrt(var + eps)) + beta
def pre_norm_transformer_block(x, params, num_heads):
    x = np.asarray(x, dtype=float)
    if isinstance(num_heads, dict):  # 兼容参数顺序倒置
        params, num_heads = num_heads, params
    if num_heads is None or 'num_heads' in params:
        num_heads = params.get('num_heads', num_heads)
    batch, seq_len, emb_dim = x.shape
    d_head = emb_dim // num_heads
    h = layer_norm(x, params['ln1_gamma'], params['ln1_beta'], eps=1e-5)
    Q = (h @ params['W_q']).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K = (h @ params['W_k']).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V = (h @ params['W_v']).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    scores = scores + causal_mask
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    context = (attn_weights @ V).transpose(0, 2, 1, 3).reshape(batch, seq_len, emb_dim)
    h = context @ params['W_o']
    x = x + h
    h = layer_norm(x, params['ln2_gamma'], params['ln2_beta'], eps=1e-5)
    ffn_h = gelu(h @ params['W_ff1'] + params['b_ff1'])
    h = ffn_h @ params['W_ff2'] + params['b_ff2']
    x = x + h
    return x