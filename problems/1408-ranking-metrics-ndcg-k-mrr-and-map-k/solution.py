import math


def dcg_at_k(rels, k):
    """计算 DCG@k；不做四舍五入。"""
    cutoff = max(0, min(k, len(rels)))

    return float(sum(
        rels[i] / math.log2(i + 2)
        for i in range(cutoff)
    ))


def ndcg_at_k(rels, k):
    """计算 NDCG@k，结果保留 4 位小数。"""
    actual_dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(rels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_rels, k)

    if ideal_dcg == 0:
        return 0.0

    return round(actual_dcg / ideal_dcg, 4)


def mrr(ranked_lists):
    """计算多条排序结果的平均倒数排名，结果保留 4 位小数。"""
    if len(ranked_lists) == 0:
        return 0.0

    reciprocal_ranks = []

    for rels in ranked_lists:
        reciprocal_rank = 0.0

        for rank, relevance in enumerate(rels, start=1):
            if relevance > 0:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def map_at_k(ranked_lists, k):
    """计算 MAP@k，结果保留 4 位小数。"""
    if len(ranked_lists) == 0:
        return 0.0

    k = max(0, k)
    average_precisions = []

    for rels in ranked_lists:
        total_relevant = sum(relevance > 0 for relevance in rels)

        # 没有相关项目，或 k=0 时，该查询的 AP 记为 0
        denominator = min(k, total_relevant)
        if denominator == 0:
            average_precisions.append(0.0)
            continue

        hits = 0
        precision_sum = 0.0
        cutoff = min(k, len(rels))

        for rank, relevance in enumerate(rels[:cutoff], start=1):
            if relevance > 0:
                hits += 1
                precision_at_rank = hits / rank
                precision_sum += precision_at_rank

        average_precisions.append(precision_sum / denominator)

    return round(sum(average_precisions) / len(average_precisions), 4)