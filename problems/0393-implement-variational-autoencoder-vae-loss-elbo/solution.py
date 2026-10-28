import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, 
             mu: np.ndarray, log_var: np.ndarray):
    """
    计算 VAE 负证据下界损失 (Negative ELBO Loss)
    
    参数:
        x: 真实原始输入，形状 (batch_size, features)
        x_reconstructed: 解码器重建输出，形状 (batch_size, features)
        mu: 潜变量均值向量，形状 (batch_size, latent_dim)
        log_var: 潜变量对数方差 ln(sigma^2)，形状 (batch_size, latent_dim)
        
    返回:
        (total_loss, reconstruction_loss, kl_divergence): 包含三个浮点数的元组
    """
    batch_size = x.shape[0]
    
    # 1. 重建损失: 计算各特征误差平方和，除以 batch_size 取得样本均值
    reconstruction_loss = np.sum((x - x_reconstructed) ** 2) / batch_size
    
    # 2. KL 散度: 闭式高斯 KL 散度求和，除以 batch_size 取得样本均值
    kl_divergence = -0.5 * np.sum(1.0 + log_var - mu**2 - np.exp(log_var)) / batch_size
    
    # 3. 负 ELBO 总损失
    total_loss = reconstruction_loss + kl_divergence
    
    return float(total_loss), float(reconstruction_loss), float(kl_divergence)