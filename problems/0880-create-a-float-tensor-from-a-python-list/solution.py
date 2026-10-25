import torch

def to_float_tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)