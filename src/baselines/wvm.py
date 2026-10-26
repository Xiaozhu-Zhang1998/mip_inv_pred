"""Python translation of the WVM method based on the Julia implementation.

Reference: https://github.com/astrzalk/WVM_reproducibility
Original Julia code: Alexander Strzalkowski (MIT); see LICENSES/WVM-MIT.
"""

import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize
import copy
from typing import List


# ---------------------------------------------------------------------------
# DirectWassersteinVariance
# ---------------------------------------------------------------------------

class DirectWassersteinVariance:
    """
    Direct one-dimensional Wasserstein variance objective.

    Parameters
    ----------
    X    : np.ndarray (n_total, d)  – predictors (all environments stacked)
    y    : np.ndarray (n_total,)    – response
    n_es : list[int]                – number of observations per environment
    w    : np.ndarray               – weights per environment (must sum to 1)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_es: List[int], w: np.ndarray):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        w = np.array(w, dtype=float)
        n_es = list(n_es)

        assert X.shape[0] == y.shape[0] == sum(n_es)
        assert abs(w.sum() - 1) <= 1e-10, "Weights must sum to 1"

        self.X = X
        self.y = y
        self.n_es = n_es
        self.w = w
        self.E = len(n_es)

        # Cumulative environment boundaries (0-based starts/ends)
        self.E_inds = np.concatenate([[0], np.cumsum(n_es)])

        # Build common probability grid (barycenter support)
        E_inds = self.E_inds
        E = self.E
        n_total = E_inds[E]

        temp_vect = np.zeros(n_total)
        for e in range(E):
            start, end = E_inds[e], E_inds[e + 1]
            ne = n_es[e]
            temp_vect[start:end] = np.arange(1, ne + 1) / ne

        pi_vect = np.unique(temp_vect)
        n = len(pi_vect)
        self.delta_pi = np.concatenate([[pi_vect[0]], np.diff(pi_vect)])

        # For each quantile level and each environment, which (sorted) obs index?
        # shape: (n, E), 0-based
        self.idx = np.zeros((n, E), dtype=int)
        for e in range(E):
            ne = n_es[e]
            self.idx[:, e] = np.ceil(pi_vect * ne).astype(int) - 1  # 0-based
            self.idx[:, e] = np.clip(self.idx[:, e], 0, ne - 1)

    def _get_res_sup_idx(self, beta: np.ndarray):
        """
        Compute residuals, barycenter support, and per-env sorted indices.
        Returns (res, supp, idx_)
        """
        E_inds = self.E_inds
        E = self.E
        n = len(self.delta_pi)

        res = self.y - self.X @ beta          # (n_total,)
        supp = np.zeros(n)
        idx_ = self.idx.copy()               # (n, E)

        for e in range(E):
            start, end = E_inds[e], E_inds[e + 1]
            res_e = res[start:end]
            perm = np.argsort(res_e)          # sort residuals in env e
            idx_[:, e] = perm[idx_[:, e]]    # map quantile ranks to sorted positions
            supp += self.w[e] * res[start + idx_[:, e]]

        return res, supp, idx_

    def __call__(self, beta: np.ndarray) -> float:
        """Evaluate the Wasserstein variance at beta."""
        res, supp, idx_ = self._get_res_sup_idx(beta)
        E_inds = self.E_inds
        wass_var = 0.0
        for e in range(self.E):
            start = E_inds[e]
            temp1 = supp - res[start + idx_[:, e]]
            wass_var += self.w[e] * float(temp1 @ (temp1 * self.delta_pi))
        return wass_var

    def gradient(self, beta: np.ndarray) -> np.ndarray:
        """Compute gradient of the Wasserstein variance at beta."""
        res, supp, idx_ = self._get_res_sup_idx(beta)
        E_inds = self.E_inds
        grad = np.zeros(len(beta))
        for e in range(self.E):
            start = E_inds[e]
            e_inds = start + idx_[:, e]
            temp1 = supp - res[e_inds]
            temp2 = temp1 * self.delta_pi
            # X.T[:, e_inds] has shape (d, n); temp2 has shape (n,)
            grad += 2 * self.w[e] * (self.X.T[:, e_inds] @ temp2)
        return grad


# ---------------------------------------------------------------------------
# Optimize WV objective
# ---------------------------------------------------------------------------

def optimize_wv(wv: DirectWassersteinVariance, initial_point: np.ndarray,
                method: str = "L-BFGS-B") -> dict:
    """
    Minimize the Wasserstein variance objective using L-BFGS-B.

    Returns a dict with keys 'x' (minimizer) and 'fun' (minimum value).
    """
    result = minimize(
        fun=wv,
        x0=initial_point,
        jac=wv.gradient,
        method=method,
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10},
    )
    return result


# ---------------------------------------------------------------------------
# Quantile KDE (Qkde)
# ---------------------------------------------------------------------------

def parabolic_kernel(u: float) -> float:
    return 0.75 * (1 - u ** 2) * (-1 <= u <= 1)


def gaussian_kernel(u: float) -> float:
    return np.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi)


class Qkde:
    """
    Kernel density estimator of the quantile density function.

    Parameters
    ----------
    eps_e   : residuals for one environment
    kernel  : kernel function (callable)
    h       : bandwidth scaling factor
    """

    def __init__(self, eps_e: np.ndarray, kernel, h: float):
        n_e = len(eps_e)
        sorted_res = np.sort(eps_e)
        self.residual_diffs = np.diff(sorted_res)
        self.kernel = kernel
        self.h = h * n_e ** (-1 / 3)
        self.n_e = n_e

    def __call__(self, x: float) -> float:
        n_e = self.n_e
        h = self.h
        quantile_points = np.arange(1, n_e) / n_e  # (n_e-1,)
        kernel_vect = np.array([self.kernel((x - q) / h) for q in quantile_points])
        return float(np.dot(self.residual_diffs, kernel_vect) / h)


def cov_kde(kde: Qkde, ss: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Covariance matrix evaluated at probability vectors ss and ts."""
    p = len(ss)
    s_mat = np.tile(ss[np.newaxis, :], (p, 1))   # (p, p)
    t_mat = np.tile(ts[:, np.newaxis], (1, p))   # (p, p)
    kde_ss = np.array([kde(s) for s in ss])
    kde_ts = np.array([kde(t) for t in ts])
    return (np.minimum(s_mat, t_mat) - np.outer(ss, ts)) * np.outer(kde_ss, kde_ts)


def calculate_moments(kde: Qkde, nsim: int):
    """Estimate mean and variance of the Wasserstein variance asymptotic dist via MC."""
    sim = np.random.uniform(0, 1, nsim)
    C = cov_kde(kde, sim, sim)
    m = np.trace(C) / nsim
    C_nodiag = C - np.diag(np.diag(C))
    cst = 2 / nsim / (nsim - 1)
    v = cst * np.linalg.norm(C_nodiag) ** 2
    return m, v


def get_asympt_dist(eps: np.ndarray, n_es: List[int], kernel, h: float, nsim: int):
    """
    Fit a Gamma distribution to the asymptotic null distribution of the WV statistic.

    Returns scipy.stats.gamma frozen distribution.
    """
    E = len(n_es)
    E_inds = np.concatenate([[0], np.cumsum(n_es)])
    n = E_inds[-1]

    kdes = [Qkde(eps[E_inds[e]:E_inds[e + 1]], kernel, h) for e in range(E)]

    def q_hat(x):
        return sum((n_es[e] / n) * kdes[e](x) for e in range(E))

    m, v = calculate_moments(q_hat, nsim)
    m *= (E - 1)
    v *= (E - 1)

    shape = m ** 2 / v
    scale = v / m
    return gamma(a=shape, scale=scale)


# ---------------------------------------------------------------------------
# Bootstrap estimation of WV null distribution
# ---------------------------------------------------------------------------

def bootstrap_wv(wv: DirectWassersteinVariance, B: int = 200,
                 method: str = "L-BFGS-B"):
    """
    Bootstrap the global minimum of the WV objective to estimate the null Gamma dist.
    Returns a scipy.stats.gamma frozen distribution.
    """
    glob_wv_min = np.zeros(B)
    n_es = wv.n_es
    E = wv.E
    E_inds = wv.E_inds
    X, y = wv.X, wv.y
    n = E_inds[-1]

    for b in range(B):
        new_X = X.copy()
        new_y = y.copy()
        for e in range(E):
            start, end = E_inds[e], E_inds[e + 1]
            ne = n_es[e]
            idx_sel = np.random.choice(np.arange(start, end), size=ne, replace=True)
            new_X[start:end] = X[idx_sel]
            new_y[start:end] = y[idx_sel]

        new_wv = DirectWassersteinVariance(new_X, new_y, n_es, np.array(n_es) / n)
        initial = np.linalg.lstsq(new_X, new_y, rcond=None)[0]
        res = optimize_wv(new_wv, initial, method=method)
        glob_wv_min[b] = res.fun

    m = np.mean(glob_wv_min)
    v = np.var(glob_wv_min, ddof=1)
    shape = m ** 2 / v
    scale = v / m
    return gamma(a=shape, scale=scale)


# ---------------------------------------------------------------------------
# WV_screening – main variable selection procedure
# ---------------------------------------------------------------------------

def WV_screening(
    wv: DirectWassersteinVariance,
    alpha=0.05,
    method: str = "L-BFGS-B",
    B: int = 200,
    nonlinear: bool = False,
    intercept: bool = True,
    N: int = 8,
):
    """
    Wasserstein Variance screening for causal variable selection.

    Parameters
    ----------
    wv        : DirectWassersteinVariance object
    alpha     : significance level (scalar or list)
    method    : scipy optimizer method
    B         : bootstrap samples for threshold estimation (0 = skip bootstrap)
    nonlinear : whether to use spline features
    intercept : include intercept in spline basis
    N         : number of spline knots per variable

    Returns
    -------
    wvvals_vect  : np.ndarray (p,) – WV value when each variable is removed
    causes_vect  : bool array or list of bool arrays indicating which vars are causal
    global_min   : float – global minimum WV value
    pvals        : np.ndarray (p,) – p-values
    thresholds   : scalar or list – threshold(s) used
    """
    X, y = wv.X, wv.y
    p = X.shape[1]

    tmp = copy.deepcopy(wv)

    if not nonlinear:
        initial_point = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        tmp.X = simple_splines(X, intercept=intercept, N=N)
        initial_point = np.linalg.lstsq(tmp.X, y, rcond=None)[0]

    # Global optimum
    res = optimize_wv(tmp, initial_point, method=method)
    opt_beta = res.x
    global_min = res.fun

    # Per-variable WV values (leaving one out)
    wvvals_vect = np.zeros(p)
    for i in range(p):
        select = [j for j in range(p) if j != i]
        wv_sub = copy.deepcopy(wv)

        if not nonlinear:
            wv_sub.X = X[:, select]
            res_i = optimize_wv(wv_sub, opt_beta[select], method=method)
        else:
            if intercept:
                inds = list(range((i) * N + 2, i * N + N + 2))
            else:
                inds = list(range(i * N + 1, i * N + N + 1))
            keep = [j for j in range(tmp.X.shape[1]) if j not in inds]
            wv_sub.X = tmp.X[:, keep]
            res_i = optimize_wv(wv_sub, opt_beta[keep], method=method)

        wvvals_vect[i] = res_i.fun

    # Bootstrap threshold
    if B > 0:
        distr = bootstrap_wv(tmp, B=B, method=method)
    else:
        # Fallback: use asymptotic distribution
        eps_hat = y - tmp.X @ opt_beta
        distr = get_asympt_dist(eps_hat, wv.n_es, gaussian_kernel, 1.0, 1000)

    # Handle scalar or list alpha
    scalar_alpha = np.isscalar(alpha)
    alphas = [alpha] if scalar_alpha else list(alpha)

    thresholds = [float(distr.ppf(1 - a)) for a in alphas]
    pvals = np.array([float(1 - distr.cdf(v)) for v in wvvals_vect])
    causes_vect = [wvvals_vect > t for t in thresholds]

    if scalar_alpha:
        return wvvals_vect, causes_vect[0], global_min, pvals, thresholds[0]
    return wvvals_vect, causes_vect, global_min, pvals, thresholds
