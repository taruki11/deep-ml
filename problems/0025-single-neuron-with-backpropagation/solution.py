import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):

	X = np.asarray(features, dtype=float)
	y = np.asarray(labels, dtype=float)
	w = np.asarray(initial_weights, dtype=float)
	b = float(initial_bias)
	lr = float(learning_rate)
	n = len(y)

	mse_values = []

	for _ in range(epochs):
		# 1. 前向传播
		z = X @ w + b
		y_hat = 1.0 / (1.0 + np.exp(-z))

		# 2. 记录当前参数下的 MSE（必须在权重更新前记录）
		mse = np.mean((y_hat - y) ** 2)
		mse_values.append(round(float(mse), 4))

		dz = (2.0 / n) * (y_hat - y) * y_hat * (1.0 - y_hat)
		dw = X.T @ dz
		db = np.sum(dz)

    # 4. 梯度下降参数更新
		w -= lr * dw
		b -= lr * db

	updated_weights = [round(float(val), 4) for val in w]
	updated_bias = round(float(b), 4)

	return updated_weights, updated_bias, mse_values