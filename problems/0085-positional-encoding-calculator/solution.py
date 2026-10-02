import numpy as np

def pos_encoding(position: int, d_model: int):
    """
    计算 Transformer 原版正弦/余弦位置编码矩阵。
    
    参数:
        position: 序列长度 (序列中有多少个 token 位置)
        d_model: 嵌入特征维度 (必须为偶数)
    返回:
        np.ndarray (dtype=float16), 形状为 (position, d_model)；若参数非法返回 -1
    """
    # 边界条件检查：长度为 0 或负数，维度非正数，直接返回 -1
    if position <= 0 or d_model <= 0:
        return -1
    
    # 1. 构造位置坐标序列: 列向量 (position, 1)，取值 [0, 1, ..., position - 1]
    pos = np.arange(position, dtype=np.float32)[:, np.newaxis]
    
    # 2. 构造特征通道偶数索引: 2i = [0, 2, 4, ..., d_model - 2]
    i = np.arange(0, d_model, 2, dtype=np.float32)
    
    # 3. 计算衰减项 (分母): 10000^(2i / d_model)
    # 为保证数值稳定与速度，使用 exp( - 2i/d_model * ln(10000) ) 等价形式
    div_term = np.exp(-i * (np.log(10000.0) / d_model))
    
    # 4. 计算每个位置在各个频率通道上的弧度角: (position, d_model // 2)
    angles = pos * div_term
    
    # 5. 交错填充正弦与余弦
    pe = np.zeros((position, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles)  # 偶数索引列 (0, 2, 4, ...)
    pe[:, 1::2] = np.cos(angles)  # 奇数索引列 (1, 3, 5, ...)
    
    # 按题目要求转为 float16
    return pe.astype(np.float16)