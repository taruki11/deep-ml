import numpy as np

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):

    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"

    # 1. 更新速度项：v_t = beta * v_{t-1} + lr * grad
    velocity = momentum * velocity + learning_rate * grad
    
    # 2. 更新参数项：theta_t = theta_{t-1} - v_t
    parameter = parameter - velocity

    # 平台判题要求保留 5 位小数并支持标量与数组统一处理
    return np.round(parameter, 5), np.round(velocity, 5)