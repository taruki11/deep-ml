import numpy as np

def baseline_forecasts(series: list[float], season: int, horizon: int) -> dict[str, list[float]]:
    n = len(series)
    last_val = float(series[-1])
    
    # 1. Naive: 重复最后一次观测值
    naive = [last_val] * horizon
    
    # 2. Seasonal Naive: 映射回倒数第一个完整周期对应位置
    start_idx = n - season
    seasonal_naive = [
        float(series[start_idx + (h % season)])
        for h in range(horizon)
    ]
    
    return {
        "naive": naive,
        "seasonal_naive": seasonal_naive
    }


def mae(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return float(sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true))