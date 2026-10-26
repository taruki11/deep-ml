# --- cell 1 ---
import numpy as np

class NeuralNetwork:
    """
    纯 NumPy 实现的两层神经网络 (784 -> 128 -> 10)
    """
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr
        
        # 1. 权重初始化（He / Kaiming 初始化，保持梯度方差稳定）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # 缓存前向传播中间结果，用于反向传播求导
        self.cache = {}
    
    def forward(self, X):
        """
        前向传播：
        Layer 1: z1 = X @ W1 + b1, h1 = ReLU(z1)
        Layer 2: z2 = h1 @ W2 + b2, probs = Softmax(z2)
        """
        # 第一层：线性变换 + ReLU 激活
        self.cache['X'] = X
        z1 = X @ self.W1 + self.b1
        self.cache['z1'] = z1
        h1 = np.maximum(0, z1)
        self.cache['h1'] = h1
        
        # 第二层：线性变换 + 数值稳定版 Softmax
        z2 = h1 @ self.W2 + self.b2
        self.cache['z2'] = z2
        
        # 减去行最大值防止 exp 产生上溢 (Numerical Stability)
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def backward(self, X, y, probs):
        """
        反向传播：手动链式法则求导并更新参数
        """
        N = X.shape[0]
        
        # 1. 计算交叉熵损失 (Cross-Entropy Loss)
        correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-12)
        loss = np.sum(correct_logprobs) / N
        
        # 2. Softmax + 交叉熵的联合导数 dL/dz2 = (probs - y_one_hot) / N
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1.0
        dscores /= N
        
        # 3. 反向传播通过第二层线性层
        dW2 = self.cache['h1'].T @ dscores
        db2 = np.sum(dscores, axis=0, keepdims=True)
        dh1 = dscores @ self.W2.T
        
        # 4. 反向传播通过 ReLU 激活函数 (导数在 z1 > 0 时为 1，否则为 0)
        dz1 = dh1 * (self.cache['z1'] > 0)
        
        # 5. 反向传播通过第一层线性层
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # 6. SGD 参数原地更新
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        return loss
    
    def train_step(self, X, y):
        """单步 mini-batch 训练：前向 -> 反向 -> 返回 loss"""
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        """推理预测类别"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)