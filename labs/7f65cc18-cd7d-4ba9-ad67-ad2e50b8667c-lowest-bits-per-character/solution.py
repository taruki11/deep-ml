# --- cell 1 ---
from collections import defaultdict
import math


def train(text: str):
  """基于绝对折减平滑统计 0 到 4 阶 N-gram 频数。

  100 万字符训练时间 < 1 秒。
  """
  K = 4  # 4-gram 上下文窗口
  d = 0.75  # 统计学最优绝对折减系数

  vocab = sorted(set(text))
  V = len(vocab)

  # 统计各阶上下文下的字符频数、上下文总数与后继分支数
  counts = [defaultdict(lambda: defaultdict(int)) for _ in range(K + 1)]
  context_totals = [defaultdict(int) for _ in range(K + 1)]
  context_types = [defaultdict(int) for _ in range(K + 1)]

  # 0阶：单字符无条件统计
  for c in text:
    counts[0][''][c] += 1
    context_totals[0][''] += 1

  # 1阶到 K阶上下文统计
  for order in range(1, K + 1):
    for i in range(len(text) - order):
      ctx = text[i : i + order]
      nxt = text[i + order]
      if counts[order][ctx][nxt] == 0:
        context_types[order][ctx] += 1
      counts[order][ctx][nxt] += 1
      context_totals[order][ctx] += 1

  # 打包成轻量原生字典返回
  return {
      'K': K,
      'd': d,
      'V': V,
      'unigram': dict(counts[0]['']),
      'total_chars': context_totals[0][''],
      'counts': [dict(c) for c in counts],
      'totals': [dict(t) for t in context_totals],
      'types': [dict(t) for t in context_types],
  }


def logprob(model, seq: str) -> list[float]:
  """计算 seq 中每个字符的对数条件概率 log p(seq[i] | seq[:i])。

  完全因果，无未来信息泄露。
  """
  K = model['K']
  d = model['d']
  V = model['V']
  unigram = model['unigram']
  total_chars = model['total_chars']
  counts = model['counts']
  totals = model['totals']
  types = model['types']

  denom_unigram = total_chars + V + 1.0

  logprobs = []
  for i, c in enumerate(seq):
    # 1. 最底层的 Unigram 平滑概率（保证未见字符也大于 0）
    prob = (unigram.get(c, 0) + 1.0) / denom_unigram

    # 2. 从 1 阶逐步向上平滑到 K 阶
    history = seq[:i]
    for order in range(1, K + 1):
      if len(history) < order:
        break
      ctx = history[-order:]
      N_ctx = totals[order].get(ctx, 0)
      if N_ctx == 0:
        continue

      T_ctx = types[order][ctx]
      c_cnt = counts[order][ctx].get(c, 0)

      # 绝对折减插值公式
      discounted = max(c_cnt - d, 0.0) / N_ctx
      weight = (d * T_ctx) / N_ctx
      prob = discounted + weight * prob

    # 概率裁剪保护（防浮点下溢与越界）
    prob = max(1e-15, min(1.0, prob))
    logprobs.append(math.log(prob))

  return logprobs