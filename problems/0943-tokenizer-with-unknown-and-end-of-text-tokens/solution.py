import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == 'encode':
        pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [p.strip() for p in pieces if p and p.strip()]
        unk_id = vocab['<|unk|>']
        return [vocab.get(token, unk_id) for token in tokens]
    elif mode == 'decode':
        inv_vocab = {v: k for k, v in vocab.items()}
        raw_text = " ".join(inv_vocab[idx] for idx in text)
        decoded_text = re.sub(r'\s+([,.?!"()\':;])', r'\1', raw_text)
        return decoded_text