"""Least squares min ||X w - y||^2 solved two ways.

Normal equations:  (X^T X) w = X^T y, solved with a Cholesky factorisation.
    Simple and fast, but cond(X^T X) = cond(X)^2, so accuracy is lost when X is ill-conditioned.
QR decomposition:   X = Q R, then R w = Q^T y.
    Works with X directly, so the error grows like cond(X) instead of cond(X)^2.
"""

import numpy as np

from .linalg import apply_qt, back_substitution, cholesky, forward_substitution, householder_qr


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so that w[0] is the intercept."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = X.T @ X
    c = X.T @ y
    L = cholesky(A)
    return back_substitution(L.T, forward_substitution(L, c))


def fit_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    reflectors, R = householder_qr(X)
    qty = apply_qt(reflectors, y)
    return back_substitution(R, qty[: X.shape[1]])


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w
