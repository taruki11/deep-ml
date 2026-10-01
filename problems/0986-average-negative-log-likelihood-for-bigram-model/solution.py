import numpy as np
import math

def bigram_avg_nll(P, words: list[str], stoi: dict) -> float:
    P = np.asarray(P)
    total_log_likelihood = 0.0
    n_bigrams = 0
    for w in words:
        padded = '.' + w + '.'
        # 用上一题讲过的 zip(padded, padded[1:]) 遍历相邻字符对
        for c1, c2 in zip(padded, padded[1:]):
            p = P[stoi[c1], stoi[c2]]
            total_log_likelihood += np.log(p)
            n_bigrams += 1
            
    # 如果没有统计到任何 bigram，返回 0.0
    if n_bigrams == 0:
        return 0.0
        
    avg_nll = -total_log_likelihood / n_bigrams
    return round(avg_nll, 4)