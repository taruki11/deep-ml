import numpy as np

def train_bigram_nn(xs, ys, vocab_size, lr=10.0, num_iters=100, alpha=0.0):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    N = len(xs)
    V = vocab_size
    W = np.zeros((V, V), dtype=float)
    X_enc = np.eye(V)[xs]
    for _ in range(num_iters):
        logits = X_enc @ W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        dlogits = probs.copy()
        dlogits[np.arange(N), ys] -= 1.0
        dlogits /= N
        dW = X_enc.T @ dlogits
        if alpha > 0:
            dW += (2.0 * alpha / (V * V)) * W
        W -= lr * dW
    return W.tolist()