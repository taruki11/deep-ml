import numpy as np


def forecast_metrics(y_true, y_pred, y_train, m=1):
  # 转为标准列表，避免复杂的维度问题
  y_true = list(y_true)
  y_pred = list(y_pred)
  y_train = list(y_train)

  n = len(y_true)
  N = len(y_train)

  # 计算测试集各点的绝对误差 |e_t|
  abs_errors = [abs(y_true[i] - y_pred[i]) for i in range(n)]

  # ========================================================
  # 1. 计算 MAPE: 只统计 y_true != 0 的点
  # ========================================================
  mape_terms = []
  for i in range(n):
    if y_true[i] != 0.0:
      mape_terms.append(abs_errors[i] / abs(y_true[i]))

  if len(mape_terms) > 0:
    mape = (sum(mape_terms) / len(mape_terms)) * 100.0
    mape_val = round(mape, 4)
  else:
    mape_val = None

  # ========================================================
  # 2. 计算 sMAPE: 真实值预测值都为 0 时该项计 0
  # ========================================================
  smape_terms = []
  for i in range(n):
    denom = abs(y_true[i]) + abs(y_pred[i])
    if denom == 0.0:
      smape_terms.append(0.0)
    else:
      smape_terms.append((2.0 * abs_errors[i]) / denom)

  smape = (sum(smape_terms) / n) * 100.0
  smape_val = round(smape, 4)

  # ========================================================
  # 3. 计算 MASE: 测试集 MAE / 训练集季节性差分均值
  # ========================================================
  if N <= m:
    mase_val = None
  else:
    # 计算训练集上间隔 m 步的差分绝对值
    naive_diffs = []
    for i in range(m, N):
      naive_diffs.append(abs(y_train[i] - y_train[i - m]))

    denom_mase = sum(naive_diffs) / len(naive_diffs)

    if denom_mase == 0.0:
      mase_val = None
    else:
      mae_test = sum(abs_errors) / n
      mase = mae_test / denom_mase
      mase_val = round(mase, 4)

  return {'mape': mape_val, 'smape': smape_val, 'mase': mase_val}