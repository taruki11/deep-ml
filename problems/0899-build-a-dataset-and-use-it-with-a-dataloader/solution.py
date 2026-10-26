import torch
from torch.utils.data import Dataset

class TensorPairDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self) -> int:
        # 返回样本总数（第 0 维长度）
        return len(self.X)

    def __getitem__(self, idx: int):
        # 返回指定索引处的单个 (特征, 标签) 样本对
        return self.X[idx], self.y[idx]