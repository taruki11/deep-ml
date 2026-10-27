import numpy as np

class DropoutLayer:
    """
    NumPy 实现的 Inverted Dropout 层。
    """
    def __init__(self, p: float):
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Drop probability p must be in [0, 1), got {p}")
        self.p = float(p)
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        前向传播:
        - training=True: 单次调用 np.random.binomial 生成 mask 并缩放。
        - training=False: 原样返回 x，且不改变 self.mask。
        """
        if not training:
            return x
        # 严格执行且仅调用一次 np.random.binomial(1, 1 - p, x.shape)
        # 1 代表保留，0 代表丢弃
        self.mask = np.random.binomial(1, 1.0 - self.p, x.shape)
        return (x * self.mask) / (1.0 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        反向传播:
        使用最近一次训练前向传播生成的 mask 传递梯度。
        若尚未生成 mask 则抛出 ValueError。
        """
        if self.mask is None:
            raise ValueError("No mask has been generated yet.")
        return (grad * self.mask) / (1.0 - self.p)