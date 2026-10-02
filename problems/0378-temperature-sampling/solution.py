import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
    logits = np.asarray(logits, dtype=float)
    
    if temperature <= 0.0:
        probs = np.zeros_like(logits, dtype=float)
        best_idx = np.argmax(logits)
        probs[best_idx] = 1.0
        return probs.tolist()
    
    scaled_logits = logits / temperature
    max_logit = np.max(scaled_logits)
    exp_logits = np.exp(scaled_logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    

    return probs.tolist()