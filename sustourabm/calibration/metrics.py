import numpy as np


def mape(y_true, y_pred):
    return (np.abs((y_true - y_pred) / y_true) * 100).mean()


def mae(y_true, y_pred):
    return (np.abs(y_true - y_pred)).mean()


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def r2(y_true, y_pred):
    return 1 - (((y_true - y_pred) ** 2).sum() / (
                (y_true - y_true.mean()) ** 2).sum())
