import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores=np.array(scores,dtype=float)
    shifted_scores=scores-np.max(scores)
    exp_scores=np.exp(shifted_scores)
    p=exp_scores/np.sum(exp_scores)

    return p    
