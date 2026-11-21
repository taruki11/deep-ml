import numpy as np

def selective_scan(u, delta, A, B, C, D):
    """
    Mamba S6 层顺序前向扫描参考实现 (对齐 selective_scan_ref).

    Args:
        u:     (B, L, ED) - 输入特征序列
        delta: (B, L, ED) - 时间步长 Δ, 驱动离散化与门控
        A:     (ED, N)    - 对角状态衰减矩阵
        B:     (B, L, N)  - 输入映射投影 (跨 ED 通道共享)
        C:     (B, L, N)  - 输出读取投影 (跨 ED 通道共享)
        D:     (ED,)      - 直通跳跃连接向量 (Skip Connection)

    Returns:
        y:          (B, L, ED) - 扫描输出序列
        last_state: (B, ED, N) - 最后一个时间步结束后的隐状态
    """
    u = np.asarray(u, dtype=float)
    delta = np.asarray(delta, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    batch_size, seq_len, ed = u.shape
    _, n = A.shape

    # 1. 初始化输出序列与初始状态 h_(-1) = 0
    y = np.zeros((batch_size, seq_len, ed), dtype=float)
    h = np.zeros((batch_size, ed, n), dtype=float)

    # 扩展 A 便于广播: (1, ED, N)
    A_broadcast = A[None, :, :]

    # 2. 沿序列长度 L 顺序递归 (Batch, Channel, State 全向量化广播)
    for t in range(seq_len):
        # 提取当前时间步切片并拓展对齐维度
        delta_t = delta[:, t, :, None]   # (B, ED, 1)
        u_t = u[:, t, :, None]           # (B, ED, 1)
        B_t = B[:, t, None, :]           # (B, 1, N)
        C_t = C[:, t, None, :]           # (B, 1, N)

        # (a) ZOH 离散化 A: exp(Δ * A) -> (B, ED, N)
        A_bar = np.exp(delta_t * A_broadcast)

        # (b) Euler 离散化 B * u: Δ * B * u -> (B, ED, N)
        dB_u = delta_t * B_t * u_t

        # (c) 隐状态更新: h_t = A_bar * h_(t-1) + Δ * B * u
        h = A_bar * h + dB_u

        # (d) 输出观测: y_t = sum_n(C * h) + D * u
        # C_t * h 结果为 (B, ED, N), 沿 -1 轴收缩求和得 (B, ED)
        y[:, t, :] = np.sum(C_t * h, axis=-1) + u[:, t, :] * D

    return y, h