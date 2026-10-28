import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """平滑且数值稳定的 Sigmoid 激活函数"""
    return 1.0 / (1.0 + np.exp(-x))

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 权重维度均为 (hidden_size, hidden_size + input_size)
        # 因为拼接向量 z = [h_{t-1}; x_t]，维度为 (H + D, 1)
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        
        # 偏置维度均为 (hidden_size, 1)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x: np.ndarray, initial_h=None, initial_c=None):
        """
        前向传播
        参数:
            x: 形状为 (T, input_size) 的序列
            initial_h: 初始隐状态 h_0，形状为 (hidden_size, 1) 或一维 (hidden_size,)
            initial_c: 初始细胞状态 c_0，形状为 (hidden_size, 1) 或一维 (hidden_size,)
        返回:
            outputs: 形状为 (T, hidden_size, 1) 的数组，包含各时间步的隐状态
            final_h: 最终时间步的隐状态，形状 (hidden_size, 1)
            final_c: 最终时间步的细胞状态，形状 (hidden_size, 1)
        """
        # 初始化隐状态 h 和细胞状态 c 为列向量 (hidden_size, 1)
        if initial_h is None:
            h = np.zeros((self.hidden_size, 1))
        else:
            h = np.array(initial_h, dtype=float).reshape(self.hidden_size, 1)
            
        if initial_c is None:
            c = np.zeros((self.hidden_size, 1))
        else:
            c = np.array(initial_c, dtype=float).reshape(self.hidden_size, 1)
            
        outputs = []
        T = len(x)
        
        for t in range(T):
            # 将当前输入整理为列向量 (input_size, 1)
            x_t = x[t].reshape(self.input_size, 1)
            
            # 将上一步的隐状态 h 与当前输入 x_t 沿垂直方向拼接
            z = np.vstack([h, x_t])  # 形状: (hidden_size + input_size, 1)
            
            # 1. 遗忘门
            f = sigmoid(np.dot(self.Wf, z) + self.bf)
            # 2. 输入门
            i = sigmoid(np.dot(self.Wi, z) + self.bi)
            # 3. 候选细胞状态
            c_bar = np.tanh(np.dot(self.Wc, z) + self.bc)
            # 4. 更新细胞状态 (长记忆更新)
            c = f * c + i * c_bar
            # 5. 输出门
            o = sigmoid(np.dot(self.Wo, z) + self.bo)
            # 6. 计算当前步隐状态 (短记忆更新)
            h = o * np.tanh(c)
            
            outputs.append(h)
            
        return np.array(outputs), h, c