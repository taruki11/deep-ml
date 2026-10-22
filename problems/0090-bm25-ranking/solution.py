import math
from collections import Counter

import numpy as np


def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if len(corpus) == 0 or len(query) == 0:
        raise ValueError("corpus and query must not be empty")

    n_docs = len(corpus)
    doc_counts = [Counter(doc) for doc in corpus]
    doc_lengths = [len(doc) for doc in corpus]
    avgdl = sum(doc_lengths) / n_docs

    # nt：包含该词的文档数；同一篇文档无论出现几次，只计一次
    document_frequency = {}
    for term in set(query):
        document_frequency[term] = sum(
            term in counts for counts in doc_counts
        )

    scores = []

    for counts, doc_length in zip(doc_counts, doc_lengths):
        score = 0.0

        # 按 query 中的词逐个计算。
        # 如果 query 中同一个词出现多次，就按出现次数重复加分。
        for term in query:
            freq = counts[term]

            # 该文档没有这个查询词，对分数没有贡献
            if freq == 0:
                continue

            nt = document_frequency[term]
            idf = math.log((n_docs + 1) / (nt + 1))

            length_norm = 1 - b + b * doc_length / avgdl

            tf_part = (
                freq * (k1 + 1)
                / (freq + k1 * length_norm)
            )

            score += idf * tf_part

        scores.append(score)

    return np.round(np.array(scores, dtype=float), 3)