# --- cell 1 ---
import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    岭回归 (L2 正则化) 闭式解 + 鲁棒特征截断 + 验证集动态寻优
    """
    # 1. 特征鲁棒截断：抑制 3 次多项式离群值产生的千万级特征爆炸
    clip_val = 5.0
    X_tr_c = np.clip(X_train, -clip_val, clip_val)
    X_val_c = np.clip(X_val, -clip_val, clip_val)
    
    n, d = X_tr_c.shape
    # 拼接全 1 偏置列
    X_tr_b = np.column_stack([np.ones(n), X_tr_c])
    X_val_b = np.column_stack([np.ones(len(X_val_c)), X_val_c])
    
    # 目标值合理边界
    y_min, y_max = np.min(y_train), np.max(y_train)
    y_pad = 0.1 * (y_max - y_min)
    low, high = y_min - y_pad, y_max + y_pad
    
    best_r2 = -np.inf
    best_theta = None
    
    # 2. 网格搜索最佳 L2 正则化强度 alpha
    for alpha in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
        I = np.eye(d + 1)
        I[0, 0] = 0.0  # 偏置项 b 不加 L2 惩罚
        
        # 求解正规方程闭式解: theta = (X^T X + alpha * I)^(-1) X^T y
        theta = np.linalg.solve(X_tr_b.T @ X_tr_b + alpha * I, X_tr_b.T @ y_train)
        
        # 验证集评估 R^2
        val_preds = np.clip(X_val_b @ theta, low, high)
        ss_res = np.sum((y_val - val_preds) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        
        if r2 > best_r2:
            best_r2 = r2
            best_theta = theta
            
    # 3. 闭包返回预测函数
    def predict(X):
        X_c = np.clip(X, -clip_val, clip_val)
        X_b = np.column_stack([np.ones(X_c.shape[0]), X_c])
        return np.clip(X_b @ best_theta, low, high)
        
    return predict