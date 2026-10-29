import numpy as np


def contrastive_loss(embeddings, temperature):
  """计算 NT-Xent / SimCLR / InfoNCE 对比损失。

  输入:
      embeddings: 形状为 (2N, d) 的 numpy array，连续对 (0,1), (2,3)... 互为正样本
      temperature: 温度系数 tau (正浮点数)
  输出:
      float: 整个批次 2N 个锚点的平均对比损失
  """
  embeddings = np.asarray(embeddings, dtype=float)
  num_views = embeddings.shape[0]

  # 1. L2 归一化 (防止除零截断)
  norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
  norms = np.maximum(norms, 1e-12)
  z = embeddings / norms

  # 2. 计算成对缩放余弦相似度矩阵: 形状 (2N, 2N)
  sim_matrix = (z @ z.T) / temperature

  # 3. 提取每个锚点的正样本相似度
  # 正样本索引利用位异或运算: 0^1=1, 1^1=0, 2^1=3, 3^1=2...
  pos_indices = np.arange(num_views) ^ 1
  pos_sim = sim_matrix[np.arange(num_views), pos_indices]

  # 4. 屏蔽自身对角线 (锚点与自身的相似度不计入分母)
  # 将对角线置为 -inf，使得 exp(-inf) = 0，且在求 max 时自动被忽略
  sim_matrix_no_self = sim_matrix.copy()
  np.fill_diagonal(sim_matrix_no_self, -np.inf)

  # 5. 采用 Log-Sum-Exp 技巧计算分母的对数和
  row_max = np.max(sim_matrix_no_self, axis=1, keepdims=True)
  log_sum_exp = row_max + np.log(
      np.sum(np.exp(sim_matrix_no_self - row_max), axis=1, keepdims=True)
  )

  # 6. 计算每个锚点的损失: loss_i = -pos_sim_i + log_sum_exp_i
  losses = -pos_sim + log_sum_exp.squeeze(1)

  # 7. 全批次求均值
  return float(np.mean(losses))


# 兼容平台可能调用的其它常用命名
nt_xent_loss = contrastive_loss
simclr_loss = contrastive_loss
infonce_loss = contrastive_loss