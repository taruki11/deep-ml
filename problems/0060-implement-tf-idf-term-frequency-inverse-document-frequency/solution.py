import math
from collections import Counter


def compute_tf_idf(corpus, query):
    n_docs = len(corpus)

    # 只需要统计查询词的 DF。
    # DF 数的是“有多少篇文档包含这个词”，不是总出现次数。
    query_words = set(query)
    document_frequency = {word: 0 for word in query_words}

    for document in corpus:
        words_in_document = set(document)
        for word in query_words:
            if word in words_in_document:
                document_frequency[word] += 1

    result = []

    for document in corpus:
        word_counts = Counter(document)
        document_length = len(document)
        scores = []

        for word in query:
            # 空文档没有词频；没出现过的查询词分数为 0
            if document_length == 0 or document_frequency[word] == 0:
                score = 0.0
            else:
                tf = word_counts[word] / document_length
                df = document_frequency[word]
                idf = math.log((n_docs + 1) / (df + 1)) + 1
                score = tf * idf

            scores.append(round(score, 5))

        result.append(scores)

    return result