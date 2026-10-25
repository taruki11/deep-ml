# --- cell 1 ---
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def train(X_unlabeled, X_labeled, y_labeled, X_val, y_val):
  """基于聚类的半监督标签传播 pipeline:

  1. 联合聚类 (K=50)
  2. 簇内多数投票传播伪标签
  3. 下游 Logistic 回归训练
  """
  K = 50
  X_all = np.vstack([X_labeled, X_unlabeled])
  n_lab = len(X_labeled)

  # 1. 对标注与无标注数据统一进行聚类
  kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
  kmeans.fit(X_all)

  # 2. 检查每个簇中落入的标注样本，用多数表决作为该簇的伪标签
  cluster_labels = {}
  for k in range(K):
    in_cluster = kmeans.labels_[:n_lab] == k
    if in_cluster.any():
      # 取落在该簇内的标注样本的众数类别
      cluster_labels[k] = int(np.bincount(y_labeled[in_cluster]).argmax())

  # 3. 仅保留能够可靠赋予伪标签的样本
  keep_indices = [
      i for i in range(len(X_all)) if kmeans.labels_[i] in cluster_labels
  ]
  X_pseudo = X_all[keep_indices]
  y_pseudo = np.array([cluster_labels[kmeans.labels_[i]] for i in keep_indices])

  # 4. 在扩增后的伪标签集上训练下游分类器
  clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
  clf.fit(X_pseudo, y_pseudo)

  def predict(X):
    return clf.predict(X)

  return predict