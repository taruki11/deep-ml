import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
  # 对输入做安全裁剪，防止极端负值下 np.exp(-z) 触发上溢警告
  z_safe = np.clip(z, -500.0, 500.0)
  return 1.0 / (1.0 + np.exp(-z_safe))


def _softplus(z: np.ndarray) -> np.ndarray:
  # 等价恒等式 log(1 + exp(z)) = max(z, 0) + log1p(exp(-|z|))，全程指数项非正，绝不溢出
  return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
  q = np.asarray(q, dtype=float)
  k = np.asarray(k, dtype=float)
  v = np.asarray(v, dtype=float)
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  g = np.asarray(g, dtype=float)
  rms_weight = np.asarray(rms_weight, dtype=float)

  T, d = q.shape

  # 1. 沿特征维度做 L2 归一化: x / (||x||_2 + eps)
  q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)
  k_norm = k / (np.linalg.norm(k, axis=-1, keepdims=True) + eps)

  # 2. 计算各时间步的衰减门 alpha_t 与更新门 beta_t
  alpha = np.exp(-_softplus(a) * np.exp(A_log))  # (T,)
  beta = _sigmoid(b)  # (T,)

  # 3. 维护固定尺寸 (d, d) 的记忆矩阵 S 并逐帧执行 Delta 更新
  S = np.zeros((d, d), dtype=float)
  outputs = np.zeros((T, d), dtype=float)

  for t in range(T):
    # a. 全局衰减遗忘
    S = alpha[t] * S
    # b. 计算预测误差 (Delta)
    delta_t = (v[t] - S.T @ k_norm[t]) * beta[t]
    # c. 外积增量写入记忆矩阵
    S = S + np.outer(k_norm[t], delta_t)
    # d. 利用归一化 Query 从当前记忆读取输出
    outputs[t] = S.T @ q_norm[t]

  # 4. 逐时间步应用 RMSNorm: (o_t / sqrt(mean(o_t^2) + eps)) * rms_weight
  rms = np.sqrt(np.mean(outputs**2, axis=-1, keepdims=True) + eps)
  o_norm = (outputs / rms) * rms_weight

  # 5. 应用 SiLU 输出门控并格式化返回原生嵌套列表
  y = o_norm * (g * _sigmoid(g))
  return np.round(y, 4).tolist()