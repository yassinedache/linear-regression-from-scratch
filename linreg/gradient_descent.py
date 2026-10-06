"""Gradient descent for the least-squares loss f(w) = (1/n) ||X w - y||^2.

Gradient:  grad f(w) = (2/n) X^T (X w - y)
Hessian:   H = (2/n) X^T X  (constant, symmetric positive semi-definite)

With L = lambda_max(H) and mu = lambda_min(H):
  - a fixed step 0 < eta < 2 / L makes the iteration converge;
  - eta > 2 / L makes it diverge;
  - the best fixed step is eta* = 2 / (L + mu), with linear rate (kappa - 1) / (kappa + 1),
    where kappa = L / mu is the condition number of H.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    r = X @ w - y
    return float(r @ r / len(y))


def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 / len(y) * (X.T @ (X @ w - y))


def hessian_spectrum(X: np.ndarray):
    """Smallest and largest eigenvalues (mu, L) of H = (2/n) X^T X."""
    eig = np.linalg.eigvalsh(2.0 / X.shape[0] * (X.T @ X))
    return float(eig[0]), float(eig[-1])


@dataclass
class GDResult:
    w: np.ndarray
    losses: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    diverged: bool = False


def gradient_descent(X: np.ndarray, y: np.ndarray, step="optimal", n_iter: int = 1000, tol: float = 1e-10,
                     w0: Optional[np.ndarray] = None) -> GDResult:
    """Batch gradient descent.

    step: a positive number (fixed step), "optimal" (2 / (L + mu)) or "armijo" (backtracking line search).
    Stops when ||grad|| < tol, when n_iter is reached, or when the loss blows up.
    """
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    if step == "optimal":
        mu, L = hessian_spectrum(X)
        eta = 2.0 / (L + mu)
    elif step == "armijo":
        eta = None
    else:
        eta = float(step)
    f = loss(w, X, y)
    res = GDResult(w, [f])
    for k in range(n_iter):
        g = gradient(w, X, y)
        if np.linalg.norm(g) < tol:
            res.converged = True
            break
        if eta is None:  # Armijo backtracking: shrink t until f decreases enough
            t = 1.0
            while loss(w - t * g, X, y) > f - 0.5 * t * (g @ g):
                t *= 0.5
            w = w - t * g
        else:
            w = w - eta * g
        f = loss(w, X, y)
        res.losses.append(f)
        if not np.isfinite(f) or f > 1e12 * (res.losses[0] + 1):
            res.diverged = True
            break
    res.w, res.iterations = w, len(res.losses) - 1
    return res
