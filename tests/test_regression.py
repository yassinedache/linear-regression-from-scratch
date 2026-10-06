import numpy as np
import pytest

from linreg import data, gradient_descent as gd, least_squares as ls, metrics


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    X = ls.add_intercept(rng.normal(size=(500, 6)))
    w_true = rng.normal(size=7)
    y = X @ w_true + 0.1 * rng.normal(size=500)
    return X, y, np.linalg.lstsq(X, y, rcond=None)[0]


def test_normal_equations_and_qr_match_numpy(problem):
    X, y, ref = problem
    assert np.allclose(ls.fit_normal_equations(X, y), ref, atol=1e-10)
    assert np.allclose(ls.fit_qr(X, y), ref, atol=1e-10)


@pytest.mark.parametrize("step", ["optimal", "armijo"])
def test_gradient_descent_reaches_the_least_squares_solution(problem, step):
    X, y, ref = problem
    res = gd.gradient_descent(X, y, step=step, n_iter=5000, tol=1e-9)
    assert res.converged
    assert np.allclose(res.w, ref, atol=1e-7)


def test_step_above_2_over_L_diverges(problem):
    X, y, _ = problem
    _, L = gd.hessian_spectrum(X)
    assert gd.gradient_descent(X, y, step=2.1 / L, n_iter=3000, tol=0).diverged
    assert not gd.gradient_descent(X, y, step=1.9 / L, n_iter=3000, tol=0).diverged


def test_observed_rate_matches_theory():
    # columns on different scales so that kappa is large enough to see the linear rate
    rng = np.random.default_rng(5)
    X = ls.add_intercept(rng.normal(size=(400, 4)) * np.array([1.0, 2.0, 4.0, 8.0]))
    y = X @ rng.normal(size=5) + 0.1 * rng.normal(size=400)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    mu, L = gd.hessian_spectrum(X)
    f_star = gd.loss(ref, X, y)
    res = gd.gradient_descent(X, y, step=1 / L, n_iter=400, tol=0)
    gap = np.array(res.losses) - f_star
    k = np.arange(gap.size)
    keep = (k > 20) & (gap > 1e-12)
    rho_obs = np.exp(np.polyfit(k[keep], np.log(gap[keep]), 1)[0] / 2)
    assert rho_obs == pytest.approx(1 - mu / L, abs=5e-3)


def test_qr_is_more_accurate_than_normal_equations_when_ill_conditioned():
    x = np.linspace(0, 1, 100)
    V = np.vander(x, 11, increasing=True)
    w_true = np.ones(11)
    y = V @ w_true
    err_ne = np.linalg.norm(ls.fit_normal_equations(V, y) - w_true)
    err_qr = np.linalg.norm(ls.fit_qr(V, y) - w_true)
    assert err_qr < 1e-3 * err_ne


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.mse(y, y) == 0
    assert metrics.r2(y, y) == 1
    assert metrics.rmse(y, y + 1) == pytest.approx(1.0)
    assert metrics.mae(y, y - 2) == pytest.approx(2.0)


def test_california_matches_scikit_learn():
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y, _ = data.load_california()
    Xtr, Xte, ytr, yte = data.train_test_split(X, y)
    s = data.Standardizer().fit(Xtr)
    w = ls.fit_qr(ls.add_intercept(s.transform(Xtr)), ytr)
    m = sklearn.LinearRegression().fit(s.transform(Xtr), ytr)
    assert w[0] == pytest.approx(m.intercept_, abs=1e-8)
    assert np.allclose(w[1:], m.coef_, atol=1e-8)
