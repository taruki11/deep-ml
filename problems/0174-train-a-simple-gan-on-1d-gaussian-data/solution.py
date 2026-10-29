import numpy as np


def sigmoid(x):
  """带截断的数值稳定 Sigmoid 函数"""
  return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def train_gan(
    mean_real,
    std_real,
    latent_dim=1,
    hidden_dim=16,
    learning_rate=0.001,
    epochs=5000,
    batch_size=128,
    seed=42,
):
  """在 1D 高斯数据上训练简单 GAN，返回生成器前向推理函数。"""
  # 1. 严格遵守题目要求的全局随机种子与权重初始化顺序
  np.random.seed(seed)
  W1_G = np.random.normal(0, 0.01, (latent_dim, hidden_dim))
  W2_G = np.random.normal(0, 0.01, (hidden_dim, 1))
  W1_D = np.random.normal(0, 0.01, (1, hidden_dim))
  W2_D = np.random.normal(0, 0.01, (hidden_dim, 1))

  # 所有偏置初始化为零 (采用 2D 行向量保持广播安全)
  b1_G = np.zeros((1, hidden_dim))
  b2_G = np.zeros((1, 1))
  b1_D = np.zeros((1, hidden_dim))
  b2_D = np.zeros((1, 1))

  # 2. 训练迭代循环
  for _ in range(epochs):
    # ---------- Step A: 判别器更新 (Discriminator Step) ----------
    # 按照严格规定抽取真实样本并重塑为 (B, 1)，再抽取噪声 z
    x_real = np.random.normal(mean_real, std_real, batch_size).reshape(
        batch_size, 1
    )
    z = np.random.normal(0, 1, (batch_size, latent_dim))

    # 生成假样本 (冻结 G，不求梯度)
    h1_G = z @ W1_G + b1_G
    a1_G = np.maximum(0, h1_G)
    x_fake = a1_G @ W2_G + b2_G

    # 判别器前向 (分别计算 real 与 fake)
    h1_D_real = x_real @ W1_D + b1_D
    a1_D_real = np.maximum(0, h1_D_real)
    logits_real = a1_D_real @ W2_D + b2_D
    D_real = sigmoid(logits_real)

    h1_D_fake = x_fake @ W1_D + b1_D
    a1_D_fake = np.maximum(0, h1_D_fake)
    logits_fake = a1_D_fake @ W2_D + b2_D
    D_fake = sigmoid(logits_fake)

    # 判别器损失关于输出 Logits 的解析导数 (带 batch 均值)
    dlogits_real = (D_real - 1.0) / batch_size
    dlogits_fake = D_fake / batch_size

    # 反向传播计算 D 的权重与偏置梯度
    grad_W2_D = a1_D_real.T @ dlogits_real + a1_D_fake.T @ dlogits_fake
    grad_b2_D = np.sum(dlogits_real, axis=0, keepdims=True) + np.sum(
        dlogits_fake, axis=0, keepdims=True
    )

    da1_D_real = dlogits_real @ W2_D.T
    dh1_D_real = da1_D_real * (h1_D_real > 0)
    da1_D_fake = dlogits_fake @ W2_D.T
    dh1_D_fake = da1_D_fake * (h1_D_fake > 0)

    grad_W1_D = x_real.T @ dh1_D_real + x_fake.T @ dh1_D_fake
    grad_b1_D = np.sum(dh1_D_real, axis=0, keepdims=True) + np.sum(
        dh1_D_fake, axis=0, keepdims=True
    )

    # 一阶普通梯度下降更新 D
    W1_D -= learning_rate * grad_W1_D
    b1_D -= learning_rate * grad_b1_D
    W2_D -= learning_rate * grad_W2_D
    b2_D -= learning_rate * grad_b2_D

    # ---------- Step B: 生成器更新 (Generator Step) ----------
    # 抽取全新的随机噪声 z_fresh
    z_fresh = np.random.normal(0, 1, (batch_size, latent_dim))

    # 生成器前向传播
    h1_G = z_fresh @ W1_G + b1_G
    a1_G = np.maximum(0, h1_G)
    x_gen = a1_G @ W2_G + b2_G

    # 输入到刚更新完毕的判别器 D
    h1_D = x_gen @ W1_D + b1_D
    a1_D = np.maximum(0, h1_D)
    logits_gen = a1_D @ W2_D + b2_D
    D_gen = sigmoid(logits_gen)

    # 非饱和生成器损失导数: d/ds [-log D] = (D - 1) / B
    dlogits_gen = (D_gen - 1.0) / batch_size

    # 梯度反向穿过判别器回传到 x_gen
    da1_D = dlogits_gen @ W2_D.T
    dh1_D = da1_D * (h1_D > 0)
    dx_gen = dh1_D @ W1_D.T

    # 生成器内部反向传播
    grad_W2_G = a1_G.T @ dx_gen
    grad_b2_G = np.sum(dx_gen, axis=0, keepdims=True)

    da1_G = dx_gen @ W2_G.T
    dh1_G = da1_G * (h1_G > 0)
    grad_W1_G = z_fresh.T @ dh1_G
    grad_b1_G = np.sum(dh1_G, axis=0, keepdims=True)

    # 一阶普通梯度下降更新 G
    W1_G -= learning_rate * grad_W1_G
    b1_G -= learning_rate * grad_b1_G
    W2_G -= learning_rate * grad_W2_G
    b2_G -= learning_rate * grad_b2_G

  # 3. 构造并返回闭包推理函数
  def gen_forward(z_in):
    """生成器前向推理。

    返回: (x_gen, a1, h1)
    """
    z_arr = np.asarray(z_in, dtype=float)
    h1 = z_arr @ W1_G + b1_G
    a1 = np.maximum(0, h1)
    x_out = a1 @ W2_G + b2_G
    return x_out, a1, h1

  return gen_forward