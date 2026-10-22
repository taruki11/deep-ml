import numpy as np
from fractions import Fraction


def random_forest_fit_predict(
    X_train,
    y_train,
    X_test,
    n_trees=5,
    max_depth=3,
    max_features=2,
    seed=0
):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)

    n_samples, n_features = X_train.shape

    if n_samples == 0:
        raise ValueError("X_train must not be empty")
    if n_trees <= 0:
        raise ValueError("n_trees must be positive")

    # 全部树共享同一个随机数生成器，保证抽样顺序符合题意
    rng = np.random.RandomState(seed)

    def gini_exact(labels):
        """用分数精确计算 Gini，避免浮点误差影响平局判定。"""
        _, counts = np.unique(labels, return_counts=True)
        n = len(labels)
        sum_squared_counts = sum(int(c) ** 2 for c in counts)
        return Fraction(n * n - sum_squared_counts, n * n)

    def make_leaf(labels):
        classes, counts = np.unique(labels, return_counts=True)
        # np.unique 的类别顺序是升序；argmax 平票时取最小类别
        prediction = int(classes[np.argmax(counts)])
        return {"leaf": True, "prediction": prediction}

    def grow_tree(X_node, y_node, depth):
        # 先检查不需要继续分裂的情况；这些叶节点不抽候选特征
        if (
            np.unique(y_node).size == 1
            or len(y_node) < 2
            or depth >= max_depth
        ):
            return make_leaf(y_node)

        # 每个可尝试分裂的节点都抽一次特征子集。
        # 此调用顺序必须是：根、完整左子树、完整右子树。
        feature_count = min(max_features, n_features)
        candidate_features = rng.choice(
            n_features,
            size=feature_count,
            replace=False
        )

        best_impurity = None
        best_feature = None
        best_threshold = None
        n_node = len(y_node)

        # 按随机抽到的特征顺序扫描
        for feature_value in candidate_features:
            feature = int(feature_value)
            values = np.unique(X_node[:, feature])

            # 阈值是相邻不同取值的中点，values 已经升序排列
            for i in range(len(values) - 1):
                lower = float(values[i])
                upper = float(values[i + 1])
                threshold = float(lower + (upper - lower) / 2.0)

                left_mask = X_node[:, feature] <= threshold
                y_left = y_node[left_mask]
                y_right = y_node[~left_mask]

                # 忽略会产生空孩子的切分
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                weighted_gini = (
                    Fraction(len(y_left), n_node) * gini_exact(y_left)
                    + Fraction(len(y_right), n_node) * gini_exact(y_right)
                )

                # 严格小于才更新；相等时保留先遇到的切分
                if best_impurity is None or weighted_gini < best_impurity:
                    best_impurity = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        # 候选特征都无法产生有效切分
        if best_feature is None:
            return make_leaf(y_node)

        left_mask = X_node[:, best_feature] <= best_threshold

        # 明确先建完整左子树，再建右子树，符合随机数调用顺序
        left_tree = grow_tree(
            X_node[left_mask],
            y_node[left_mask],
            depth + 1
        )
        right_tree = grow_tree(
            X_node[~left_mask],
            y_node[~left_mask],
            depth + 1
        )

        return {
            "leaf": False,
            "feature": int(best_feature),
            "threshold": float(best_threshold),
            "left": left_tree,
            "right": right_tree,
        }

    def predict_one(tree, row):
        while not tree["leaf"]:
            feature = tree["feature"]
            threshold = tree["threshold"]

            if row[feature] <= threshold:
                tree = tree["left"]
            else:
                tree = tree["right"]

        return int(tree["prediction"])

    trees = []

    for _ in range(n_trees):
        # 每棵树先抽 bootstrap 样本
        bootstrap_indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]

        # 再按递归顺序抽每个节点的候选特征
        trees.append(grow_tree(X_bootstrap, y_bootstrap, depth=0))

    # 每一行是一棵树对所有测试样本的预测
    tree_predictions = np.asarray([
        [predict_one(tree, row) for row in X_test]
        for tree in trees
    ])

    # 逐个测试样本投票；np.unique 升序排列，所以平票时选最小标签
    final_predictions = []
    for column in tree_predictions.T:
        classes, counts = np.unique(column, return_counts=True)
        final_predictions.append(int(classes[np.argmax(counts)]))

    return final_predictions