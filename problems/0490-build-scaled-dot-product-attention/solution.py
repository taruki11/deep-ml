import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
	d_k=K.shape[-1]
	scores=(Q @ K.T)/np.sqrt(d_k)
	if mask is not None:
		mask=np.asarray(mask,dtype=float)
		scores=np.where(mask==0,-1e9,scores)
	scores_max=np.max(scores,axis=-1,keepdims=True)
	exp_scores = np.exp(scores - scores_max)
	attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
	output = attention_weights @ V
	return output, attention_weights