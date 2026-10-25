import torch

def grad_of_quadratic(x_value) -> float:
    # 1. 确保输入为浮点数并构造开启 requires_grad 的叶子张量
    x = torch.tensor(float(x_value), requires_grad=True)
    
    # 2. 前向计算 f(x) = x^2 + 3x + 2，Autograd 自动记录计算图
    y = x ** 2 + 3 * x + 2
    
    # 3. 对标量结果触发反向传播
    y.backward()
    
    # 4. 从叶子张量的 grad 属性中提取标量 Python float
    return float(x.grad.item())
    