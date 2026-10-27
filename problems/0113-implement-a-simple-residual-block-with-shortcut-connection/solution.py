import torch

def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:

    # y = ReLU(w2 @ ReLU(w1 @ x) + x)
    h1 = torch.relu(torch.matmul(w1, x))
    h2 = torch.matmul(w2, h1)
    return torch.relu(h2 + x)