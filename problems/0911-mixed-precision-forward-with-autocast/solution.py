import torch
import torch.nn as nn

def forward_bf16(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    使用 torch.amp.autocast 执行 Bfloat16 混合精度前向传播。
    
    参数:
        model (nn.Module): 神经网络模型 (参数通常存储为 float32)
        x (torch.Tensor): 输入张量，数据类型为 float32
        
    返回:
        torch.Tensor: 输出张量，类型被安全转换回 float32
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)

    # 动态适配当前张量所在设备 ('cpu' 或 'cuda')
    device_type = x.device.type

    # 使用 PyTorch 跨设备统一 AMP 接口
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(x)

    # 在模型边界处转回 float32，避免下游 Loss 函数或其余管线出现低精度类型不兼容
    return output.float()