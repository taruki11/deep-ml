import os
import warnings

# 屏蔽底层 C++ 警告输出
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 1. 第一层卷积：3x3，步长为 stride，padding 为 1，不带偏置
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        # 2. 第一层批归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3. 第二层卷积：3x3，步长固定为 1，padding 为 1，不带偏置
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        # 4. 第二层批归一化
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 5. 残差连接（Shortcut）
        # 当通道一致且 stride=1 时使用恒等映射；否则使用 1x1 Conv + BatchNorm 投影对齐
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out