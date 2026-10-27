import torch
import torch.nn as nn
from typing import List, Tuple, Any

def accumulated_step(
    model: nn.Module,
    micro_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: Any
) -> float:
    """
    通过微批次梯度累加执行一次优化器更新。
    
    参数:
        model: PyTorch 模型 (nn.Module)
        micro_batches: 微批次数据列表 [(x1, y1), (x2, y2), ...]
        optimizer: 优化器 (如 torch.optim.SGD / AdamW)
        criterion: 损失函数 (如 nn.MSELoss / nn.CrossEntropyLoss)
        
    返回:
        float: 所有微批次未缩放的真实损失平均值 (Python float)
    """
    # 1. 在所有微批次计算前，仅清空一次梯度
    optimizer.zero_grad()

    num_micro_batches = len(micro_batches)
    if num_micro_batches == 0:
        return 0.0

    total_unscaled_loss = 0.0

    # 2. 依次前向并反向传播每个微批次
    for x, y in micro_batches:
        outputs = model(x)
        loss = criterion(outputs, y)

        # 记录未缩放的真实损失 (用于最后计算真实平均 Loss)
        total_unscaled_loss += loss.item()

        # 核心步骤: 将当前微批次的 loss 缩小 N 倍后进行反向传播
        # PyTorch 的 backward 会在 param.grad 上原地累加 (+=)
        scaled_loss = loss / num_micro_batches
        scaled_loss.backward()

    # 3. 所有微批次累加完毕后，严格执行且仅执行一次优化器步进
    optimizer.step()

    # 4. 返回未缩放真实损失的算术平均值 (原生 Python float)
    mean_unscaled_loss = total_unscaled_loss / num_micro_batches
    return float(mean_unscaled_loss)