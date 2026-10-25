import torch
import torch.nn as nn

def train_one_step(model, x, y, lr: float) -> float:
    # 1. 针对模型的参数构造局部 SGD 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 2. 清空历史累加梯度
    optimizer.zero_grad()
    
    # 3. 前向传播计算预测值
    pred = model(x)
    
    # 4. 计算均方误差（MSE）损失，并记录更新前的 loss 标量值
    loss = nn.MSELoss()(pred, y)
    loss_val = float(loss.item())
    
    # 5. 反向传播计算各参数梯度
    loss.backward()
    
    # 6. 优化器单步更新参数
    optimizer.step()
    
    return loss_val