import numpy as np

def rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b):
    # 1. 统一转为 NumPy 浮点数组
    h = np.array(initial_hidden_state, dtype=float)
    Wx = np.array(Wx, dtype=float)
    Wh = np.array(Wh, dtype=float)
    b = np.array(b, dtype=float)
    
    # 2. 顺着时间线，遍历每一个时间步的输入向量
    for x in input_sequence:
        x = np.array(x, dtype=float)
        # 3. 核心计算：输入变换 + 隐状态循环变换 + 偏置，过 tanh 激活
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
        
    # 4. 保留 4 位小数返回最终记忆状态
    return np.round(h, 4).tolist()