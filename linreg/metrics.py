"""Regression metrics."""

import numpy as np


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def rmse(y, p):
    return float(np.sqrt(mse(y, p)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
