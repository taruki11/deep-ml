import numpy as np

class SimpleRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # 严格遵守题目要求的权重生成顺序与随机数规范
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # 缓存正向传播状态，供反向传播 BPTT 使用
        self.last_inputs = None
        self.last_hidden_states = None
        self.last_outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        正向传播
        参数:
            x: 形状为 (T, input_size) 的序列数组
        返回:
            outputs: 形状为 (T, output_size, 1) 的预测数组
        """
        T = x.shape[0]
        h_t = np.zeros((self.hidden_size, 1))
        
        # 记录所有的隐状态，h_list[0] 为初始 h_0 = 0，h_list[t] 为 step t 的隐状态
        h_states = [h_t]
        x_list = []
        outputs = []
        
        for t in range(T):
            # 将 x[t] 转换为列向量 (input_size, 1)
            x_t = x[t].reshape(self.input_size, 1)
            x_list.append(x_t)
            
            # 循环隐状态方程: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            linear_h = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_t) + self.b_h
            h_t = np.tanh(linear_h)
            h_states.append(h_t)
            
            # 线性输出方程: y_t = W_hy * h_t + b_y
            y_t = np.dot(self.W_hy, h_t) + self.b_y
            outputs.append(y_t)
            
        # 缓存中间值，供 backward 反向传播使用
        self.last_inputs = x_list
        self.last_hidden_states = h_states
        self.last_outputs = outputs
        
        return np.array(outputs)

    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float):
        """
        反向传播 (BPTT) 并更新参数
        参数:
            x: 形状为 (T, input_size) 的输入序列
            y: 形状为 (T, output_size) 的目标标签
            learning_rate: 学习率
        """
        T = x.shape[0]
        
        # 初始化 5 个参数的梯度累加器（与参数维度完全相同）
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 来自未来步的梯度 dh_next，在最后一步 T 时未来没有节点，初始为 0
        dh_next = np.zeros((self.hidden_size, 1))
        
        # 沿时间轴从后往前反向遍历: t = T-1, T-2, ..., 0
        for t in reversed(range(T)):
            y_pred = self.last_outputs[t]               # 形状: (output_size, 1)
            y_target = y[t].reshape(self.output_size, 1) # 形状: (output_size, 1)
            h_curr = self.last_hidden_states[t + 1]     # 当前步隐状态 h_t
            h_prev = self.last_hidden_states[t]         # 上一步隐状态 h_{t-1}
            x_t = self.last_inputs[t]                   # 当前步输入 x_t
            
            # 1. 损失函数 L = 0.5 * ||y_pred - y_target||^2 的导数
            dy = y_pred - y_target                      # 形状: (output_size, 1)
            
            # 2. 累积输出层参数的梯度
            dW_hy += np.dot(dy, h_curr.T)               # (output_size, 1) @ (1, hidden_size) -> (output_size, hidden_size)
            db_y += dy                                  # (output_size, 1)
            
            # 3. 隐状态 h_t 接收到的两路梯度求和
            dh = np.dot(self.W_hy.T, dy) + dh_next      # (hidden_size, 1)
            
            # 4. 穿过 tanh 激活层: d/dz tanh(z) = 1 - tanh^2(z)
            dz = (1.0 - h_curr ** 2) * dh               # (hidden_size, 1)
            
            # 5. 累积循环层参数梯度
            dW_xh += np.dot(dz, x_t.T)                  # (hidden_size, 1) @ (1, input_size) -> (hidden_size, input_size)
            dW_hh += np.dot(dz, h_prev.T)               # (hidden_size, 1) @ (1, hidden_size) -> (hidden_size, hidden_size)
            db_h += dz                                  # (hidden_size, 1)
            
            # 6. 计算沿时间向过去传递的梯度，供 t-1 步使用
            dh_next = np.dot(self.W_hh.T, dz)           # (hidden_size, 1)
            
        # 7. 普通梯度下降更新: theta <- theta - lr * grad
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y