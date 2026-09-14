import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, eps = 1e-15) -> float:
  # 1. 对预测概率做 epsilon 截断，防止 log(0)
    p_clipped = np.clip(predicted_probs, eps, 1 - eps)

    # 2. 逐元素相乘并按行求和算出每个样本的损失
    # 形状变化: (N, C) -> (N,)
    sample_losses = -np.sum(true_labels * np.log(p_clipped), axis=-1)

    # 3. 对整个批次求平均值
    return float(np.mean(sample_losses))