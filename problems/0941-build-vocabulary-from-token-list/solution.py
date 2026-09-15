from collections import Counter
def build_vocab(tokens):
    return {token: idx for idx, token in enumerate(sorted(set(tokens)))}
    
    
