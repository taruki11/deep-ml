def bpe_decode(ids, vocab):
    if not ids:
        return ""
    result = []
    for tokens_id in ids:
        token=vocab[tokens_id]
        if token.startswith('G'):
            result.append(' '+token[1:])
        else:
            result.append(token)
    return ''.join(result)
    pass