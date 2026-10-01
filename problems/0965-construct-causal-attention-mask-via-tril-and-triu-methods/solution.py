import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    T = attn_weights.shape[0]
    if method == 'tril':
        mask = np.tril(np.ones((T, T)))
        masked = attn_weights * mask
        res = masked / np.sum(masked, axis=-1, keepdims=True)
    elif method == 'triu':
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        
        scores = np.log(attn_weights)
        scores[mask] = -np.inf
        

        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        res = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
    else:
        raise ValueError(f"Unsupported method: {method}")
        
    return res.tolist()
