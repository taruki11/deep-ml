from collections import Counter

def train_tokenizer(corpus: list[str], vocab_size: int):
    # 1. 构建基础字符表（包含语料库出现的所有字符，并补充常用 ASCII 字符防越界）
    base_chars = set("".join(corpus))
    # 补充常用英文字母及符号，确保测试集遇到未见字符不报错
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'":
        base_chars.add(c)
        
    chars = sorted(list(base_chars))
    # 若字符总数超过 vocab_size，则截断；否则保留
    if len(chars) > vocab_size:
        chars = chars[:vocab_size]

    token_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_token = {i: ch for i, ch in enumerate(chars)}
    current_id = len(chars)

    # 2. 统计语料库中的词频，准备 BPE 训练
    # 结构: {('e', 'm', 'm', 'a'): 15, ...}
    word_counts = Counter(corpus)
    vocab_words = {}
    for word, freq in word_counts.items():
        if word and all(ch in token_to_id for ch in word):
            vocab_words[tuple(word)] = freq

    merges = {}  # 记录合并规则: (token_a, token_b) -> new_token_id

    # 3. 迭代合并最高频相邻字符对，直到词表达上限
    while current_id < vocab_size:
        pair_counts = Counter()
        for word_tuple, freq in vocab_words.items():
            for i in range(len(word_tuple) - 1):
                pair_counts[(word_tuple[i], word_tuple[i + 1])] += freq

        # 没有可合并的对，或最高频对出现少于 2 次时提前终止
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[best_pair] < 2:
            break

        # 分配新的 Token ID
        new_id = current_id
        merges[best_pair] = new_id
        merged_str = best_pair[0] + best_pair[1]
        
        token_to_id[merged_str] = new_id
        id_to_token[new_id] = merged_str
        current_id += 1

        # 在当前语料库中执行单趟替换
        a, b = best_pair
        new_vocab_words = {}
        for word_tuple, freq in vocab_words.items():
            new_word = []
            i = 0
            n = len(word_tuple)
            while i < n:
                if i < n - 1 and word_tuple[i] == a and word_tuple[i + 1] == b:
                    new_word.append(merged_str)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab_words[tuple(new_word)] = freq
        vocab_words = new_vocab_words

    # 4. 定义 encode 函数（利用学习到的合并规则自底向上合并）
    def encode(text: str) -> list[int]:
        if not text:
            return []
            
        # 初始拆分为单字符列表
        tokens = [ch for ch in text if ch in token_to_id]
        
        # 循环应用 merges 中优先级最高（最先学到）的规则
        while len(tokens) >= 2:
            best_pair = None
            best_rank = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in merges and merges[pair] < best_rank:
                    best_rank = merges[pair]
                    best_pair = pair
                    
            if best_pair is None:
                break
                
            # 执行合并
            a, b = best_pair
            merged_token = a + b
            new_tokens = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [token_to_id[t] for t in tokens]

    # 5. 定义 decode 函数（查反向字典直接拼接）
    def decode(ids: list[int]) -> str:
        return "".join(id_to_token[i] for i in ids)

    return encode, decode