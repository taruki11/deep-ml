from tinygrad import Tensor

def grad_of_quadratic(x_value):
    """
    Computes the gradient of f(x) = x^2 + 3x + 2 at x = x_value using Tinygrad's autograd.
    
    Args:
        x_value: int or float
        
    Returns:
        float: gradient f'(x_value) = 2 * x_value + 3
    """
    # 1. 转换为浮点数并构造 0-D Tensor，开启梯度追踪
    x = Tensor(float(x_value), requires_grad=True)
    
    # 2. 前向传播
    y = x**2 + 3 * x + 2
    
    # 3. 反向传播计算梯度
    y.backward()
    
    # 4. 从 x.grad 中提取标量浮点值并返回
    return float(x.grad.item())