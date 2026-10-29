# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def build_model(img_size: int = 8, n_classes: int = 2) -> nn.Module:
    """
    构建符合 Deep-ML 要求的轻量级 CNN 模型
    标准流水线: Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear
    """
    class TinyCNN(nn.Module):
        def __init__(self, img_size, n_classes):
            super().__init__()
            self.out_channels = 16
            
            # 1. 单通道卷积提取 16 组局部特征，padding=1 保持尺寸为 8x8
            self.conv = nn.Conv2d(in_channels=1, out_channels=self.out_channels, 
                                  kernel_size=3, padding=1)
            # 2. 激活函数
            self.relu = nn.ReLU()
            # 3. 2x2 最大池化，尺寸从 8x8 降为 4x4
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 4. 全连接分类层: 16 通道 * 4 高度 * 4 宽度 = 256
            pooled_dim = img_size // 2
            flatten_dim = self.out_channels * pooled_dim * pooled_dim
            self.fc = nn.Linear(flatten_dim, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)  # 展平为 (N, 256)
            x = self.fc(x)             # 映射为 (N, n_classes)
            return x

    return TinyCNN(img_size, n_classes)

def train_model(model: nn.Module, 
                train_x: torch.Tensor, 
                train_y: torch.Tensor, 
                epochs: int = 15, 
                lr: float = 0.01, 
                batch_size: int = 32, 
                seed: int = 0) -> nn.Module:
    """
    在 CPU 上原地训练模型
    """
    # 1. 固定随机种子确保结果 100% 确定
    torch.manual_seed(seed)
    
    # 2. 封装数据加载器
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. 定义损失函数与优化器 (Adam 在小样本分类任务中收敛极快)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. 训练迭代循环
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()          # 清空历史梯度
            logits = model(batch_x)         # 前向传播得到未归一化对数几率
            loss = criterion(logits, batch_y) # 计算交叉熵损失
            loss.backward()                 # 反向传播求导
            optimizer.step()                # 权重更新
            
    return model