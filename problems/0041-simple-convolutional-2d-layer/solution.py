import torch
import torch.nn.functional as F

def simple_conv2d(
    input_matrix: torch.Tensor, 
    kernel: torch.Tensor, 
    padding: int, 
    stride: int
) -> torch.Tensor:
    """
    使用 PyTorch 纯张量从底层滑窗实现 2D 卷积
    """
    if not isinstance(input_matrix, torch.Tensor):
        input_matrix = torch.as_tensor(input_matrix, dtype=torch.float32)
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.as_tensor(kernel, dtype=torch.float32)

    Kh, Kw = kernel.shape

    # F.pad 参数顺序为: (左, 右, 上, 下)
    if padding > 0:
        padded = F.pad(input_matrix, (padding, padding, padding, padding), mode='constant', value=0.0)
    else:
        padded = input_matrix

    pH, pW = padded.shape
    out_h = (pH - Kh) // stride + 1
    out_w = (pW - Kw) // stride + 1

    out = torch.zeros((out_h, out_w), dtype=input_matrix.dtype, device=input_matrix.device)

    for i in range(out_h):
        for j in range(out_w):
            window = padded[i * stride : i * stride + Kh, j * stride : j * stride + Kw]
            out[i, j] = torch.sum(window * kernel)

    return out