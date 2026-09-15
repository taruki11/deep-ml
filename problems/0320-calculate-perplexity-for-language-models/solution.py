import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    p=np.array(probabilities)
    return np.exp(-np.sum(np.log(p))/len(p))