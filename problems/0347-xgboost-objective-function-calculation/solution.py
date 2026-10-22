import numpy as np


def xgboost_objective(
    gradients,
    hessians,
    left_indices,
    right_indices,
    lambda_reg=1.0,
    gamma=0.0
):
    gradients = np.asarray(gradients, dtype=float)
    hessians = np.asarray(hessians, dtype=float)
    left_indices = np.asarray(left_indices, dtype=int)
    right_indices = np.asarray(right_indices, dtype=int)

    G_left = float(gradients[left_indices].sum())
    H_left = float(hessians[left_indices].sum())

    G_right = float(gradients[right_indices].sum())
    H_right = float(hessians[right_indices].sum())

    G_parent = G_left + G_right
    H_parent = H_left + H_right

    left_weight = -G_left / (H_left + lambda_reg)
    right_weight = -G_right / (H_right + lambda_reg)

    gain = 0.5 * (
        G_left**2 / (H_left + lambda_reg)
        + G_right**2 / (H_right + lambda_reg)
        - G_parent**2 / (H_parent + lambda_reg)
    ) - gamma

    return {
        "left_weight": round(left_weight, 4),
        "right_weight": round(right_weight, 4),
        "gain": round(gain, 4),
    }