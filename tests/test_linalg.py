import numpy as np
import pytest

from linreg import linalg


def random_matrix(m, n, seed=0):
    return np.random.default_rng(seed).normal(size=(m, n))


def test_cholesky_reconstructs_the_matrix():
    B = random_matrix(6, 6)
    A = B @ B.T + 6 * np.eye(6)
    L = linalg.cholesky(A)
    assert np.allclose(L, np.tril(L))
    assert np.allclose(L @ L.T, A)


def test_cholesky_rejects_non_positive_definite():
    with pytest.raises(np.linalg.LinAlgError):
        linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_triangular_solves():
    rng = np.random.default_rng(1)
    L = np.tril(rng.normal(size=(5, 5))) + 5 * np.eye(5)
    b = rng.normal(size=5)
    assert np.allclose(L @ linalg.forward_substitution(L, b), b)
    assert np.allclose(L.T @ linalg.back_substitution(L.T, b), b)


def test_householder_qr():
    A = random_matrix(30, 5, seed=2)
    reflectors, R = linalg.householder_qr(A)
    Q = linalg.thin_q(reflectors, 30, 5)
    assert np.allclose(R, np.triu(R))
    assert np.allclose(Q.T @ Q, np.eye(5))
    assert np.allclose(Q @ R, A)


def test_apply_qt_matches_explicit_q():
    A = random_matrix(20, 4, seed=3)
    b = np.random.default_rng(4).normal(size=20)
    reflectors, _ = linalg.householder_qr(A)
    Q = linalg.thin_q(reflectors, 20, 4)
    assert np.allclose(linalg.apply_qt(reflectors, b)[:4], Q.T @ b)
