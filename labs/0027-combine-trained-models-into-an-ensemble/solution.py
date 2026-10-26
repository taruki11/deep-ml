# --- cell 1 ---
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_ensemble(base_models, X_val, y_val):
    """
    Stacking 集成：利用基础模型预测的类别概率作为特征，训练元分类器。
    """
    K = len(base_models)
    n_val = len(y_val)

    # 1. 提取所有基础模型在验证集上的预测概率分布
    # val_probs 形状: (K, n_val, n_classes)
    val_probs = np.array([m.predict_proba(X_val) for m in base_models])

    # 2. 转换为元特征矩阵：每行对应一个样本拼接所有模型的概率
    # 形状: (n_val, K * n_classes) -> 每个样本对应 5 * 10 = 50 维元特征
    val_features = val_probs.transpose(1, 0, 2).reshape(n_val, -1)

    # 3. 在验证集特征上拟合元分类器（LogisticRegression）
    meta = LogisticRegression(max_iter=2000, C=1.0)
    meta.fit(val_features, y_val)

    # 4. 返回推理阶段的预测函数
    def predict(X):
        # 提取测试样本的基础模型概率特征
        test_probs = np.array([m.predict_proba(X) for m in base_models])
        test_features = test_probs.transpose(1, 0, 2).reshape(X.shape[0], -1)
        # 由元分类器输出最终的类别预测结果
        return meta.predict(test_features)

    return predict