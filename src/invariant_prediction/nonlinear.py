"""Nonlinear extension: basis expansion and group subset selection."""
import numpy as np

# From original notebook, cell 43
from sklearn.preprocessing import SplineTransformer

def spline_expand(X_list, degree=3, n_knots=3, knots="quantile", standardize=True):
    X_list = [np.asarray(X, float) for X in X_list]
    X_pool, E, p = np.vstack(X_list), len(X_list), X_list[0].shape[1]
    Z_blocks, groups, basis_info, start = [[] for _ in range(E)], [], [], 0

    for j in range(p):
        sp = SplineTransformer(
            degree=degree, n_knots=n_knots, knots=knots, include_bias=False
        ).fit(X_pool[:, [j]])

        Zj = [sp.transform(X[:, [j]]) for X in X_list]
        Rj = Zj[0].shape[1]

        scale = np.ones(Rj)
        if standardize:
            scale = np.sqrt(np.mean(np.vstack(Zj)**2, axis=0))
            scale[scale < 1e-10] = 1.
            Zj = [Z / scale for Z in Zj]

        groups.append(np.arange(start, start + Rj))
        for e in range(E): Z_blocks[e].append(Zj[e])

        basis_info.append({"transformer": sp, "scale": scale, "Rj": Rj})
        start += Rj

    return [np.hstack(Z) for Z in Z_blocks], groups, basis_info

# From original notebook, cell 44
def fourier_expand(X_list, K=1):
    X_list = [np.asarray(X, float) for X in X_list]
    E, p = len(X_list), X_list[0].shape[1]
    Z_blocks, groups, start = [[] for _ in range(E)], [], 0

    for j in range(p):
        for e in range(E):
            x = X_list[e][:, j]
            Z_blocks[e].append(np.column_stack([
                f(k*x) for k in range(1, K+1) for f in (np.sin, np.cos)
            ]))
        Rj = 2*K
        groups.append(np.arange(start, start + Rj))
        start += Rj

    return [np.hstack(Z) for Z in Z_blocks], groups

# From original notebook, cell 45
def precompute_stat(X_list, y_list, n_list=None):
    X_list = [np.asarray(X, float) for X in X_list]
    y_list = [np.asarray(y, float).reshape(-1) for y in y_list]
    E = len(X_list)
    n = np.array([len(y) for y in y_list], float) if n_list is None else np.asarray(n_list, float)
    w = n / n.sum()
    y2 = [float(y @ y) for y in y_list]
    Xty = [X.T @ y for X, y in zip(X_list, y_list)]
    XtX = [X.T @ X for X in X_list]

    return {
        "E": E, 
        "n_list": n, 
        "w": w, 
        "w_pair": np.outer(w, w),
        "y2_list": y2, 
        "Xty_list": Xty, 
        "XtX_list": XtX,
        "Xty_pool": np.sum(Xty, axis=0), 
        "XtX_pool": np.sum(XtX, axis=0)
    }

# From original notebook, cell 46
def group_cols(S, groups):
    S = sorted(set(map(int, S)))
    if len(S) == 0:
        return np.array([], int)
    return np.sort(np.concatenate([groups[j] for j in S]))

# From original notebook, cell 47
def compute_B_given_groups(S, stat, groups, rcond=1e-10):
    A, E, R = group_cols(S, groups), stat["E"], stat["XtX_pool"].shape[0]
    B = np.zeros((E + 1, R))
    if not len(A): return B

    B[0, A] = np.linalg.pinv(
        stat["XtX_pool"][np.ix_(A, A)], rcond=rcond
    ) @ stat["Xty_pool"][A]

    for e in range(E):
        B[e + 1, A] = np.linalg.pinv(
            stat["XtX_list"][e][np.ix_(A, A)], rcond=rcond
        ) @ stat["Xty_list"][e][A]

    return B

# From original notebook, cell 48
def compute_mse_obj_given_groups(
    B, S, stat, gamma1=1., rho=1., lam=0., return_parts=False
):
    E, n, w, wp = stat["E"], stat["n_list"], stat["w"], stat["w_pair"]
    y2, Xty, XtX = stat["y2_list"], stat["Xty_list"], stat["XtX_list"]
    beta = B[0]

    pred = sum(
        w[e] * (y2[e] - 2*Xty[e]@beta + beta@XtX[e]@beta) / n[e]
        for e in range(E)
    )

    resid = [
        (y2[e] - Xty[e] @ B[e + 1]) / n[e]
        for e in range(E)
    ]

    coef_diff = var_diff = 0.
    for e in range(E):
        for f in range(e + 1, E):
            d = B[e + 1] - B[f + 1]
            coef_diff += wp[e, f] * d @ d
            var_diff += wp[e, f] * (resid[e] - resid[f])**2

    sparsity = lam * len(S)
    obj = pred + gamma1*coef_diff + gamma1*rho*var_diff + sparsity

    if not return_parts: return obj
    return obj, {
        "prediction": pred,
        "coef_diff_raw": coef_diff,
        "coef_diff": gamma1*coef_diff,
        "var_diff_raw": var_diff,
        "var_diff": gamma1*rho*var_diff,
        "sparsity": sparsity,
        "total": obj
    }

# From original notebook, cell 49
def cd_msetype_grouped(
    stat, groups, lam, gamma1, rho,
    forbid=None, S_init=(), superset_init=(),
    tol=1e-10, max_iter=100, rcond=1e-10
):
    p = len(groups)
    forbid = set() if forbid is None else set(forbid)
    S = set(S_init) - forbid
    superset = (set(superset_init) - forbid) | S

    def fit(S):
        S0 = sorted(S)
        B = compute_B_given_groups(S0, stat, groups, rcond)
        obj = compute_mse_obj_given_groups(B, S0, stat, gamma1, rho, lam)
        return B, obj

    B, obj = fit(S)

    for _ in range(max_iter):

        # Stage 1
        while True:
            improved = False
            for j in sorted(superset):
                if j in forbid: continue
                S_new = S - {j} if j in S else S | {j}
                B_new, obj_new = fit(S_new)

                if obj_new < obj - tol:
                    S, B, obj = S_new, B_new, obj_new
                    improved = True

            if not improved: break

        # Stage 2
        new_feature = False
        for j in range(p):
            if j in forbid: continue

            adding_new = j not in S and j not in superset
            S_new = S - {j} if j in S else S | {j}
            B_new, obj_new = fit(S_new)

            if obj_new < obj - tol:
                S, B, obj = S_new, B_new, obj_new
                if adding_new:
                    superset.add(j)
                    new_feature = True
                break

        if not new_feature: break

    S = sorted(map(int, S))
    return S, B, compute_mse_obj_given_groups(B, S, stat, gamma1, rho, lam)

