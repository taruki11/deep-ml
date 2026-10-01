import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    Q=np.asarray(Q,dtype=float)
    K=np.asarray(K,dtype=float)
    d_k = K.shape[-1]
    scaled_scores = (Q @ K.T) / np.sqrt(d_k)
    scores_max=np.max(scaled_scores,axis=-1,keepdims=True)
    exp_scores=np.exp(scaled_scores-scores_max)
    weights=exp_scores/np.sum(exp_scores,axis=-1,keepdims=True)
    return np.round(weights, 4).tolist()
