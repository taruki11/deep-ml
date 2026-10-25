# --- cell 1 ---
import numpy as np


class DecisionTree:

  def __init__(
      self,
      max_depth=10,
      min_samples_split=2,
      max_features='sqrt',
      random_state=None,
  ):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.max_features = max_features  # 'sqrt', None, 或整数
    self.random_state = random_state

  def fit(self, X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    self.n_classes_ = int(np.max(y)) + 1
    n_features = X.shape[1]

    # 确定每次分裂尝试的特征数量
    if self.max_features == 'sqrt':
      self.n_features_to_try = max(1, int(np.sqrt(n_features)))
    elif self.max_features is None:
      self.n_features_to_try = n_features
    else:
      self.n_features_to_try = min(int(self.max_features), n_features)

    self.rng = np.random.RandomState(self.random_state)
    self.tree = self._build(X, y, depth=0)
    return self

  def _gini(self, y):
    if len(y) == 0:
      return 0.0
    counts = np.bincount(y, minlength=self.n_classes_)
    p = counts / len(y)
    return 1.0 - float(np.sum(p**2))

  def _majority(self, y):
    return int(np.bincount(y, minlength=self.n_classes_).argmax())

  def _best_split(self, X, y):
    n_samples, n_features = X.shape
    # 核心：随机采样特征子集
    feat_idx = self.rng.choice(
        n_features, self.n_features_to_try, replace=False
    )
    best_gain, best_f, best_t = 0.0, None, None
    parent_g = self._gini(y)

    for f in feat_idx:
      values = X[:, f]
      uniq = np.unique(values)
      if len(uniq) <= 1:
        continue

      # 连续特征切分加速：最多选取 20 个等间距阈值
      if len(uniq) > 20:
        idx = np.linspace(0, len(uniq) - 1, 20).astype(int)
        cands = uniq[idx]
      else:
        cands = uniq

      for t in cands:
        left = values <= t
        n_l = int(left.sum())
        if n_l == 0 or n_l == n_samples:
          continue

        w = n_l / n_samples
        g = parent_g - w * self._gini(y[left]) - (1 - w) * self._gini(y[~left])
        if g > best_gain:
          best_gain, best_f, best_t = g, int(f), float(t)

    return best_f, best_t, best_gain

  def _build(self, X, y, depth):
    # 终止条件：达到最大深度、样本不足或纯度已达到 100%
    if (
        depth >= self.max_depth
        or len(y) < self.min_samples_split
        or len(np.unique(y)) == 1
    ):
      return {'leaf': True, 'class': self._majority(y)}

    f, t, g = self._best_split(X, y)
    if f is None or g <= 0.0:
      return {'leaf': True, 'class': self._majority(y)}

    mask = X[:, f] <= t
    return {
        'leaf': False,
        'feature': f,
        'threshold': t,
        'left': self._build(X[mask], y[mask], depth + 1),
        'right': self._build(X[~mask], y[~mask], depth + 1),
    }

  def predict(self, X):
    X = np.asarray(X, dtype=np.float64)
    out = np.empty(X.shape[0], dtype=np.int64)

    for i in range(X.shape[0]):
      node = self.tree
      while not node['leaf']:
        if X[i, node['feature']] <= node['threshold']:
          node = node['left']
        else:
          node = node['right']
      out[i] = node['class']

    return out