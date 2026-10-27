import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):

    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    # 迭代步数 t 从 1 开始（用于偏差修正的分母 1 - beta^t）
    for t in range(1, num_iterations + 1):
        g = grad(x)
        
        # 1. 更新一阶矩估计（动量）
        m = beta1 * m + (1.0 - beta1) * g
        
        # 2. 更新二阶矩估计（未中心化的方差）
        v = beta2 * v + (1.0 - beta2) * (g ** 2)
        
        # 3. 一阶偏差修正
        m_hat = m / (1.0 - beta1 ** t)
        
        # 4. 二阶偏差修正
        v_hat = v / (1.0 - beta2 ** t)
        
        # 5. 更新参数
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x