import numpy as np


def yule_walker(x, p):
  """使用尤尔-沃克方程拟合 AR(p) 模型并进行一步预测。

  参数:
      x: 一维列表或数组，时序样本数据
      p: 整数，自回归阶数 (1 <= p < len(x))

  返回:
      字典形式的参数与预测结果，保留 4 位小数
  """
  data = np.asarray(x, dtype=float).ravel()
  n = len(data)
  mean_val = float(np.mean(data))
  dev = data - mean_val

  # 1. 计算滞后 0 到 p 的样本自协方差
  gamma = np.zeros(p + 1)
  for k in range(p + 1):
    if k == 0:
      gamma[0] = np.sum(dev * dev) / n
    else:
      gamma[k] = np.sum(dev[:-k] * dev[k:]) / n

  # 边缘处理：若序列为常数（无方差）
  if np.isclose(gamma[0], 0.0):
    return {
        'phi': [0.0] * p,
        'sigma2': 0.0,
        'forecast': round(float(mean_val), 4),
    }

  # 2. 构建 Toeplitz 协方差矩阵 R 与目标向量 r
  R = np.zeros((p, p))
  for i in range(p):
    for j in range(p):
      R[i, j] = gamma[abs(i - j)]

  r = gamma[1 : p + 1]

  # 3. 求解线性方程组 R * phi = r
  phi = np.linalg.solve(R, r)

  # 计算噪声方差 sigma^2
  sigma2 = gamma[0] - float(np.dot(phi, r))
  sigma2 = max(0.0, sigma2)

  # 4. 一步预测：根据最近 p 个历史值推算下一个值
  recent_dev = dev[-1 : -p - 1 : -1]  # [x_{n-1}-mu, x_{n-2}-mu, ...]
  forecast = mean_val + float(np.dot(phi, recent_dev))

  # 统一格式化为保留 4 位小数
  return {
      'phi': [round(float(v), 4) for v in phi],
      'sigma2': round(float(sigma2), 4),
      'forecast': round(float(forecast), 4),
  }