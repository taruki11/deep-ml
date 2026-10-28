import torch
import torch.nn.functional as F

def distillation_loss(student, teacher, temperature: float = 1.0) -> torch.Tensor:
    """
    知识蒸馏损失函数 (Hinton 官方标准公式: Loss = T^2 * KL)
    
    参数:
        student: 学生模型输出 logits (Tensor 或 ndarray)
        teacher: 教师模型输出 logits (Tensor 或 ndarray)
        temperature: 温度超参数 T
    返回:
        torch.Tensor 标量损失 (支持 .item())
    """
    # 兼容传入 NumPy 数组或 PyTorch 张量
    if not isinstance(student, torch.Tensor):
        student = torch.tensor(student, dtype=torch.float32)
    if not isinstance(teacher, torch.Tensor):
        teacher = torch.tensor(teacher, dtype=torch.float32)
        
    T = float(temperature)
    
    # 1. 计算温度软化后的对数概率与目标概率
    log_p_student = F.log_softmax(student / T, dim=-1)
    p_teacher = F.softmax(teacher / T, dim=-1)
    
    # 2. 计算 KL 散度 (1D 样本用 sum，2D batch 用 batchmean)
    if student.dim() == 1:
        kl = F.kl_div(log_p_student, p_teacher, reduction='sum')
    else:
        kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
        
    # 3. 核心修复点: 乘上 T^2
    return (T ** 2) * kl