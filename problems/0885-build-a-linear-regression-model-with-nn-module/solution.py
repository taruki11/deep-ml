import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        # 1. 必须调用父类构造函数，完成 Module 内部的状态注册机制初始化
        super().__init__()
        # 2. 将 nn.Linear 赋值给 self 的属性，nn.Module 会自动将其注册为子模块
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3. 前向传播：将输入送入 self.linear
        return self.linear(x)