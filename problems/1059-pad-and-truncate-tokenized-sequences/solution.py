def pad_and_truncate(sequences, pad_token_id, max_length=None):
    if not sequences:
        return []
        
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
        
    result = []
    for seq in sequences:
        # 先切片截断（若原本比 max_length 短，切片保留原长）
        truncated = seq[:max_length]
        # 不足 max_length 的部分，在末尾用 pad_token_id 补齐
        padded = truncated + [pad_token_id] * (max_length - len(truncated))
        result.append(padded)
        
    return result
