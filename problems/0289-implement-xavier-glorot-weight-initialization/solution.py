import math
import torch


def xavier_initialization(
    shape: tuple, mode: str = "uniform", seed: int = None
) -> torch.Tensor:
  if seed is not None:
    torch.manual_seed(seed)

  fan_in, fan_out = shape[0], shape[1]
  w = torch.empty(shape)

  if mode == "uniform":
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    w.uniform_(-limit, limit)
  elif mode == "normal":
    sigma = math.sqrt(2.0 / (fan_in + fan_out))
    w.normal_(0.0, sigma)
  else:
    raise ValueError(f"Invalid mode: {mode}")

  return w