import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Sequence

def run_blocks(
    blocks: Sequence[nn.Module], 
    x: torch.Tensor, 
    use_checkpoint: bool = False
) -> torch.Tensor:
    """
    按顺序对输入张量 x 应用 blocks 中的各个模块。
    
    参数:
        blocks: 包含若干子模块的序列 (如 nn.ModuleList 或 list of Modules)
        x: 输入张量
        use_checkpoint (bool): 是否启用梯度检查点
        
    返回:
        torch.Tensor: 前向计算结果
    """
    for block in blocks:
        if use_checkpoint:
            # 必须显式指定 use_reentrant=False:
            # 1. 消除写入 stderr 的 UserWarning 告警
            # 2. 适配现代 autograd 引擎，支持更广泛的张量形态
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x