import numpy as np

def hinge_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    losses = np.maximum(0, 1 - y_true * y_pred)
    return round(float(np.mean(losses)), 4)