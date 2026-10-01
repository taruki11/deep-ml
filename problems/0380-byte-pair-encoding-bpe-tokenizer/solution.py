def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    vocab={tuple(word.split()): freq for word, freq in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = {}
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        a, b = best_pair
        merged_token = a + b
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged_token)
                    i += 2  # 合并成功，跳过两个 token
                else:
                    new_word.append(word[i])
                    i += 1  
            new_vocab[tuple(new_word)] = freq
             
        vocab = new_vocab
    return merges