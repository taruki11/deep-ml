import numpy as np

def posterior(X, y, m0, S0, sigma2):
    """
    计算贝叶斯线性回归的批量高斯后验分布参数 (mN, SN)。
    """
    S0_inv = np.linalg.inv(S0)
    # 后验精度矩阵：先验精度 + 数据精度
    SN_inv = S0_inv + (X.T @ X) / sigma2
    # 后验协方差矩阵
    SN = np.linalg.inv(SN_inv)
    # 后验均值向量
    mN = SN @ (S0_inv @ m0 + (X.T @ y) / sigma2)
    return mN, SN

def sequential_posterior(X, y, m0, S0, sigma2):
    """
    逐样本（流式）序贯更新后验分布。
    返回: (最终后验均值 mN, 最终后验协方差 SN, 历史后验均值轨迹 means)
    """
    N, D = X.shape
    cur_m = m0.copy()
    cur_S = S0.copy()
    means = np.zeros((N, D))
    
    for i in range(N):
        # 每次只传入单个样本点 (1, D) 和 (1,)
        cur_m, cur_S = posterior(X[i:i+1], y[i:i+1], cur_m, cur_S, sigma2)
        means[i] = cur_m
        
    return cur_m, cur_S, means

def predictive(X_test, mN, SN, sigma2):
    """
    计算测试集上的后验预测分布均值和方差。
    返回: (预测均值向量 mean, 预测方差向量 var)
    """
    # 预测均值: X_test @ mN
    mean = X_test @ mN
    
    # 预测方差: sigma^2 + x^T @ SN @ x
    # 利用 np.sum((X_test @ SN) * X_test, axis=1) 并行计算每行的二次型 x^T SN x
    param_uncertainty = np.sum((X_test @ SN) * X_test, axis=1)
    var = sigma2 + param_uncertainty
    
    return mean, var