import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的 Cross-Entropy 实现（匹配 F.cross_entropy 默认行为）。
    
    参数:
        logits: 形状为 (N, C) 的未归一化分数 (Float Tensor)
        targets: 形状为 (N,) 的整数类别标签 (Long Tensor)
        
    返回:
        标量 Tensor (标量损失)
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, dtype=torch.long)

    # 1. 提取每个样本真实目标类对应的 logit: z_{i, y_i}
    # 使用 gather 保持对任意设备 (CPU/GPU) 与自动求导的纯函数兼容
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 2. 计算每行的稳定 Log-Sum-Exp: log(sum(exp(logits)))
    # torch.logsumexp 内部自动执行了减 max 的平移变换，防止 exp 溢出
    log_sum_exp = torch.logsumexp(logits, dim=-1)

    # 3. 计算单样本损失: log_sum_exp - target_logit
    per_sample_loss = log_sum_exp - target_logits

    # 4. 默认 reduction='mean'
    return per_sample_loss.mean()