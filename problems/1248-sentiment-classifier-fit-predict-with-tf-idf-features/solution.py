import numpy as np
from collections import Counter


class SentimentClassifier:
    def __init__(self, learning_rate=0.5, iterations=2000, l2=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2

        self.vocab = {}
        self.idf = None
        self.weights = None
        self.bias = 0.0
        self.constant_label = None

    def _tokenize(self, text):
        # 题目规定的分词方式：转小写，再按空格切分
        return text.lower().split()

    def _vectorize(self, tokenized_texts):
        """把多篇分词后的文本转换成 TF-IDF 矩阵，形状为 (文本数, 词表大小)。"""
        n_texts = len(tokenized_texts)
        n_words = len(self.vocab)
        X = np.zeros((n_texts, n_words), dtype=float)

        for row, tokens in enumerate(tokenized_texts):
            counts = Counter(
                token for token in tokens
                if token in self.vocab
            )

            for token, count in counts.items():
                col = self.vocab[token]
                X[row, col] = count * self.idf[col]

            # L2 归一化：让不同长度文本的向量尺度更可比
            norm = np.linalg.norm(X[row])
            if norm > 0:
                X[row] /= norm

        return X

    def fit(self, texts, labels):
        labels = np.asarray(labels, dtype=float)

        if len(texts) != len(labels):
            raise ValueError("texts 和 labels 的长度必须相同")
        if len(texts) == 0:
            raise ValueError("训练数据不能为空")

        tokenized_texts = [self._tokenize(text) for text in texts]

        # 按训练文本的扫描顺序建立词表。
        # 普通 Python 字典会保留插入顺序。
        self.vocab = {}
        for tokens in tokenized_texts:
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # Document Frequency：统计每个词出现在多少篇不同文本中
        document_frequency = np.zeros(len(self.vocab), dtype=float)

        for tokens in tokenized_texts:
            for token in set(tokens):
                document_frequency[self.vocab[token]] += 1

        # 平滑 IDF，避免分母为 0
        n_docs = len(texts)
        self.idf = np.log(
            (n_docs + 1) / (document_frequency + 1)
        ) + 1

        X = self._vectorize(tokenized_texts)

        # 训练集只有一个类别时，直接始终预测这个类别
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            self.constant_label = int(unique_labels[0])
            self.weights = np.zeros(X.shape[1])
            self.bias = 0.0
            return self

        self.constant_label = None

        # 逻辑回归参数：词权重向量和截距
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0

        # 全批量梯度下降，最小化二元交叉熵并加入 L2 正则化
        n_samples = len(labels)

        for _ in range(self.iterations):
            logits = X @ self.weights + self.bias
            logits = np.clip(logits, -50, 50)

            probabilities = 1.0 / (1.0 + np.exp(-logits))
            errors = probabilities - labels

            grad_weights = (X.T @ errors) / n_samples
            grad_weights += self.l2 * self.weights
            grad_bias = np.mean(errors)

            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

        return self

    def predict(self, text):
        if self.weights is None:
            raise ValueError("请先调用 fit 训练分类器")

        if self.constant_label is not None:
            return self.constant_label

        tokens = self._tokenize(text)
        X = self._vectorize([tokens])  # 一篇文本，形状为 (1, 词表大小)

        score = X[0] @ self.weights + self.bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))

        return int(probability >= 0.5)