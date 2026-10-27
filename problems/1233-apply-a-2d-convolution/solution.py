import torch
import torch.nn as nn

def apply_conv2d() -> torch.Tensor:
    # 1. 创建 2D 卷积层：单输入通道，单输出通道，核大小 2x2，无偏置
    conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
    
    # 2. 定义固定的卷积核权重，形状为 (C_out=1, C_in=1, kH=2, kW=2)
    weight = torch.tensor([[[[1.0, 0.0],
                             [0.0, 1.0]]]], dtype=torch.float32)
    
    # 3. 在无梯度模式下将固定权重覆盖到卷积层
    with torch.no_grad():
        conv.weight.copy_(weight)
        
    # 4. 定义固定的输入张量，形状为 (B=1, C=1, H=3, W=3)
    x = torch.tensor([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]]], dtype=torch.float32)
    
    # 5. 在无梯度模式下执行前向传播
    with torch.no_grad():
        out = conv(x)
        
    return out