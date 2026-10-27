import math

class CosineAnnealingLRScheduler:
    """
    Cosine Annealing Learning Rate Scheduler.
    
    Formula:
        lr_t = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(min(t, T_max) / T_max * pi))
    """
    def __init__(self, initial_lr: float, T_max: int, min_lr: float):
        self.initial_lr = float(initial_lr)
        self.T_max = int(T_max)
        self.min_lr = float(min_lr)

    def get_lr(self, epoch: int) -> float:
        # 当 epoch 超过 T_max 时截断在 T_max 处
        current_epoch = min(epoch, self.T_max)
        
        # 余弦退火核心计算公式
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + math.cos(current_epoch / self.T_max * math.pi)
        )
        
        # 平台要求四舍五入保留 4 位小数
        return round(lr, 4)