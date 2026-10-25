import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:

  # 1. Sigmoid 导数: sigma(x) * (1 - sigma(x))
  s = 1.0 / (1.0 + np.exp(-x))
  sigmoid_deriv = float(s * (1.0 - s))

  # 2. Tanh 导数: 1 - tanh^2(x)
  t = np.tanh(x)
  tanh_deriv = float(1.0 - t**2)

  # 3. ReLU 导数: x > 0 时为 1.0，否则为 0.0
  relu_deriv = 1.0 if x > 0 else 0.0

  return {"sigmoid": sigmoid_deriv, "tanh": tanh_deriv, "relu": relu_deriv}