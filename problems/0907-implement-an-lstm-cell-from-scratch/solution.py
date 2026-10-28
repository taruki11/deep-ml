import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 标准初始化界限: k = 1 / sqrt(hidden_size)
        k = 1.0 / math.sqrt(hidden_size)
        
        # 严格遵守题目指定的变量名与形状
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size).uniform_(-k, k))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k))
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))

    def forward(self, x: torch.Tensor, state=None):
        """
        单个时间步前向传播
        参数:
            x: 形状为 (N, input_size) 的当前步输入张量 (N 为 batch_size)
            state: 元组 (h_prev, c_prev)，每个形状均为 (N, hidden_size)
        返回:
            (h_new, c_new): 包含更新后隐状态和细胞状态的元组，形状均为 (N, hidden_size)
        """
        # 若未提供初始状态，初始化为全 0
        if state is None:
            N = x.size(0)
            h_prev = torch.zeros(N, self.hidden_size, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(N, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        # 1. 单次大矩阵乘法合并计算预激活值 g: (N, 4 * hidden_size)
        g = x @ self.W_ih.t() + self.b_ih + h_prev @ self.W_hh.t() + self.b_hh
        
        # 2. 沿特征维度平分切为 4 份，每份形状 (N, hidden_size)
        # 顺序: 输入门 (i), 遗忘门 (f), 候选细胞 (c_cand), 输出门 (o)
        i, f, c_cand, o = g.chunk(4, dim=1)
        
        # 3. 门控经过 Sigmoid [0, 1]，候选细胞经过 Tanh [-1, 1]
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_tilde = torch.tanh(c_cand)
        o = torch.sigmoid(o)
        
        # 4. 更新细胞状态 (长途记忆加法累积)
        c_new = f * c_prev + i * c_tilde
        
        # 5. 更新隐状态 (短途记忆输出)
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new