# --- cell 1 ---
import numpy as np

def optimizer_step(param, grad, state, lr):
    """
    带动量的随机梯度下降 (SGD with Momentum)
    
    参数说明:
        param: 当前权重数组 (NumPy ndarray, 比如形状为 [784, 128])
        grad:  当前权重对应的梯度 (NumPy ndarray, 形状与 param 完全相同)
        state: 状态字典 (dict)，用于跨 step 存储历史累积速度/动量
        lr:    学习率标量 (float)
        
    返回值:
        new_param: 更新后的权重数组 (形状与 param 一致)
        state:     更新后的状态字典
    """
    # 1. 如果该参数是第一次更新，初始化动量缓存为全 0 矩阵
    if 'v' not in state:
        state['v'] = np.zeros_like(param)
        
    # 2. 动量衰减系数 (物理学中的惯性系数，通常取 0.9)
    beta = 0.9
    
    # 3. 累积历史动量: v = beta * v + grad
    state['v'] = beta * state['v'] + grad
    
    # 4. 沿动量方向更新参数: w_new = w - lr * v
    new_param = param - lr * state['v']
    
    return new_param, state