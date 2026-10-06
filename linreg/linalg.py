"""Linear algebra written from scratch (only NumPy arrays and basic operations).

- Cholesky factorisation A = L L^T of a symmetric positive definite matrix
- forward and back substitution for triangular systems
- Householder QR factorisation A = Q R and the least-squares solve that uses it
"""

import numpy as np


def cholesky(A: np.ndarray) -> np.ndarray:
    """Return the lower-triangular L with A = L L^T (A symmetric positive definite)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        d = A[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0:
            raise np.linalg.LinAlgError("matrix is not positive definite")
        L[j, j] = np.sqrt(d)
        L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L x = b for lower-triangular L."""
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def back_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve U x = b for upper-triangular U."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def _householder_vector(x: np.ndarray):
    """Vector v (with v[0] = 1) and beta such that (I - beta v v^T) x = -sign(x0) ||x|| e1."""
    alpha = np.linalg.norm(x)
    if alpha == 0:
        return np.zeros_like(x), 0.0
    v = x.copy()
    v[0] += np.copysign(alpha, x[0])  # avoids cancellation
    v /= v[0]
    beta = 2.0 / (v @ v)
    return v, beta


def householder_qr(A: np.ndarray):
    """Householder QR of an m x n matrix (m >= n).

    Returns the list of reflectors (v_k, beta_k) and the n x n upper-triangular R.
    Q is never formed explicitly: Q^T b is obtained by applying the reflectors.
    """
    R = np.array(A, dtype=float, copy=True)
    m, n = R.shape
    reflectors = []
    for k in range(n):
        v, beta = _householder_vector(R[k:, k])
        R[k:, k:] -= beta * np.outer(v, v @ R[k:, k:])
        reflectors.append((v, beta))
    return reflectors, np.triu(R[:n, :])


def apply_qt(reflectors, b: np.ndarray) -> np.ndarray:
    """Compute Q^T b using the stored Householder reflectors."""
    y = np.array(b, dtype=float, copy=True)
    for k, (v, beta) in enumerate(reflectors):
        y[k:] -= beta * v * (v @ y[k:])
    return y


def thin_q(reflectors, m: int, n: int) -> np.ndarray:
    """Form the m x n matrix Q with orthonormal columns (used for checks and tests)."""
    Q = np.eye(m, n)
    for k in range(len(reflectors) - 1, -1, -1):
        v, beta = reflectors[k]
        Q[k:, :] -= beta * np.outer(v, v @ Q[k:, :])
    return Q
