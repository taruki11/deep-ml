import numpy as np

def bigram_counts(words):
    # Your code here
    unique_chars = sorted(list(set("".join(words))))
    vocab = ['.'] + unique_chars
    V = len(vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    N = np.zeros((V, V), dtype=int)
    for w in words:
        augmented=['.']+list(w)+['.']
        for a,b in zip(augmented,augmented[1:]):
            N[stoi[a],stoi[b]]+=1
    return N.tolist()