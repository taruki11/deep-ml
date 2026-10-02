import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:

    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    # 1. 沿最后一个维度 (特征维度) 计算均值
    # keepdims=True 保持形状为 (batch_size, sequence_length, 1)，便于广播
    mean = np.mean(X, axis=-1, keepdims=True)
    
    # 2. 沿特征维度计算方差
    variance = np.var(X, axis=-1, keepdims=True)
    
    # 3. 标准化: 减均值，除以标准差 (加 eps 防止除以 0)
    X_hat = (X - mean) / np.sqrt(variance + eps)
    
    # 4. 仿射变换: 缩放与平移
    output = gamma * X_hat + beta
    
    return output