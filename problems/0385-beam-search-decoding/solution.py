import numpy as np


def beam_search_decode(log_probs, beam_width, top_k=1):
  log_probs = np.asarray(log_probs, dtype=float)
  T, V = log_probs.shape

  # 初始束：仅包含一条空序列和 0.0 的累计对数概率
  beams = [([], 0.0)]

  for t in range(T):
    candidates = []
    # 1. 扩展：将当前保留的每条序列与当前步所有词表 token 拼接
    for seq, score in beams:
      for v in range(V):
        new_seq = seq + [int(v)]
        new_score = score + float(log_probs[t, v])
        candidates.append((new_seq, new_score))

    # 2. 剪枝：按累计对数概率降序排列，仅保留前 beam_width 条
    candidates.sort(key=lambda x: x[1], reverse=True)
    beams = candidates[:beam_width]

  # 3. 仅在最终返回时进行四舍五入，避免中间步截断误差累积
  return [(seq, round(score, 4)) for seq, score in beams[:top_k]]