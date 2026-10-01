def bpe_encode(text, token_to_id, merges):
    if not text:
        return []
    ids=[token_to_id[ch] for ch in text]
    while len(ids)>=2:
        best_pair=None
        best_id=float('inf')
        for i in range(len(ids)-1):
            pair=(ids[i],ids[i+1])
            if pair in merges and merges[pair]<best_id:
                best_id=merges[pair]
                best_pair=pair
        if best_pair is None:
            break
        a, b = best_pair
        new_ids = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                new_ids.append(best_id)
                i += 2  # 跳过已被合并的两个 token
            else:
                new_ids.append(ids[i])
                i += 1
                    
        ids = new_ids   # Your code here
    return ids