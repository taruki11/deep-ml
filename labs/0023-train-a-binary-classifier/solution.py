# --- cell 1 ---
import numpy as np
from sklearn.linear_model import LogisticRegression


def train(X_train, y_train, X_val, y_val):
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)

    def predict(X):
        X = np.asarray(X)
        return model.predict(X).astype(int)

    return predict