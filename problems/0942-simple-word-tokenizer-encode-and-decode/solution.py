import re

def encode(text, vocab):
    pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [p.strip() for p in pieces if p and p.strip()]
    return [vocab[token] for token in tokens]

def decode(ids, vocab):
    inverse_vocab = {v: k for k, v in vocab.items()}
    tokens = [inverse_vocab[i] for i in ids]
    raw_text = " ".join(tokens)
    result = re.sub(r'\s+([,.?!"()\'])', r'\1', raw_text)
    return result