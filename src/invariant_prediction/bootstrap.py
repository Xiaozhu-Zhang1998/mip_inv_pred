"""Task 2 sample splitting, candidate supports, and multiplier bootstrap."""
import numpy as np
from itertools import combinations
from .task2 import precompute_stat_test,cd_testtype,vanilla_testtype

# From original notebook, cell 29
def train_test_split_envs(X_list, y_list, test_size=.5, seed=0, shuffle=True):
    rng = np.random.default_rng(seed)
    X_train, y_train, X_test, y_test = [], [], [], []

    for X, y in zip(X_list, y_list):
        n = len(y)
        idx = rng.permutation(n) if shuffle else np.arange(n)
        n_test = int(np.floor(test_size*n))
        te, tr = idx[:n_test], idx[n_test:]

        X_train.append(X[tr]); y_train.append(y[tr])
        X_test.append(X[te]); y_test.append(y[te])

    n_train_list = [len(y) for y in y_train]
    n_test_list = [len(y) for y in y_test]

    return X_train, y_train, n_train_list, X_test, y_test, n_test_list

# From original notebook, cell 31
def _pinv(A, rcond=1e-10):
    return np.linalg.pinv(A, rcond=rcond)

def _ols(X, y, rcond=1e-10):
    return np.zeros(0) if X.shape[1] == 0 else _pinv(X.T @ X, rcond) @ (X.T @ y)

def _pooled_ols(X_list, y_list, S, rcond=1e-10):
    S = np.asarray(S, int)
    if len(S) == 0: return np.zeros(0)
    XtX = sum(X[:, S].T @ X[:, S] for X in X_list)
    Xty = sum(X[:, S].T @ y for X, y in zip(X_list, y_list))
    return _pinv(XtX, rcond) @ Xty

# From original notebook, cell 32
def _bootstrap_cache(X_list, y_list, n_list, S, rcond=1e-10):
    S, N = np.asarray(S, int), int(np.sum(n_list))
    d = len(S)
    beta_pool = _pooled_ols(X_list, y_list, S, rcond)
    Sigma_pool, env = np.zeros((d, d)), []

    for X, y, ne in zip(X_list, y_list, n_list):
        XS = X[:, S] if d else np.zeros((ne, 0))
        beta_e = _ols(XS, y, rcond)
        re, rp = y - XS @ beta_e, y - XS @ beta_pool
        XtX = XS.T @ XS

        if d:
            Sigma_pool += XtX
            Sinv, mu, C, q = _pinv(XtX/ne, rcond), XS.mean(0), XtX/ne, XS.T @ rp/ne
        else:
            Sinv, mu, q, C = np.zeros((0,0)), np.zeros(0), np.zeros(0), np.zeros((0,0))

        env.append({
            "XS": XS, "re": re, "rp": rp, "Sinv": Sinv,
            "mu": mu, "C": C, "q": q,
            "m": float(rp.mean()), "v": float(np.mean(re**2))
        })

    Spinv = _pinv(Sigma_pool/N, rcond) if d else np.zeros((0,0))
    return {"S": S, "d": d, "Spinv": Spinv, "env": env}

# From original notebook, cell 33
def bootstrap_testtype(
    X_list, y_list, n_list, S_list, gamma=(1,1,1,1),
    B=300, random_state=None, rcond=1e-10, raw_scale=True
):
    gamma = np.asarray(gamma, float)
    if gamma.shape != (4,) or np.any(gamma < 0):
        raise ValueError("gamma must contain four nonnegative values.")

    X_list = [np.asarray(X, float) for X in X_list]
    y_list = [np.asarray(y, float).reshape(-1) for y in y_list]
    n = np.asarray(n_list, int)
    E, N = len(X_list), int(n.sum())
    w, pairs = n/N, list(combinations(range(E), 2))
    wp = np.outer(w, w)
    rng = np.random.default_rng(random_state)

    S_list = list(dict.fromkeys(tuple(sorted(map(int, S))) for S in S_list))
    if not S_list: raise ValueError("S_list cannot be empty.")
    cache = [_bootstrap_cache(X_list, y_list, n, S, rcond) for S in S_list]

    boot = np.zeros(B)

    for b in range(B):
        xi = [rng.normal(size=n[e]) for e in range(E)]
        best = -np.inf

        for c in cache:
            d, env = c["d"], c["env"]

            if d:
                score = sum(env[e]["XS"].T @ (xi[e]*env[e]["rp"]) for e in range(E)) / np.sqrt(N)
                Bstar = c["Spinv"] @ score
            else:
                Bstar = np.zeros(0)

            L, V, O, K = [], [], [], []

            for e in range(E):
                ne, sc, z, ce = n[e], np.sqrt(N/n[e]), xi[e], env[e]
                XS, re, rp = ce["XS"], ce["re"], ce["rp"]

                if d:
                    G = XS.T @ (z*re) / np.sqrt(ne)
                    Le = sc * ce["Sinv"] @ G
                    Xr = XS * rp[:, None]
                    Q = (z[:, None] * (Xr-ce["q"][None,:])).sum(0) / np.sqrt(ne)
                    Ke = sc*Q - ce["C"] @ Bstar
                else:
                    Le = Ke = np.zeros(0)

                H = np.sum(z*(re**2-ce["v"])) / np.sqrt(ne)
                M = np.sum(z*(rp-ce["m"])) / np.sqrt(ne)

                L.append(Le)
                V.append(sc*H)
                O.append(sc*M - ce["mu"] @ Bstar)
                K.append(Ke)

            t1 = t2 = t3 = 0.
            t4_each = np.zeros(d)

            for e, f in pairs:
                dL, dV, dO, dK = L[e]-L[f], V[e]-V[f], O[e]-O[f], K[e]-K[f]
                t1 += wp[e,f] * (dL @ dL)
                t2 += wp[e,f] * dV**2
                t3 += wp[e,f] * dO**2
                if d: t4_each += wp[e,f] * dK**2

            t4 = np.max(t4_each) if d else 0.
            best = max(best, gamma @ np.array([t1, t2, t3, t4]))

        boot[b] = best

    return boot/N if raw_scale else boot

# From original notebook, cell 34
def bootstrap_env_data(X_list, y_list, n_list, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Xb, yb = [], []

    for X, y, ne in zip(X_list, y_list, n_list):
        idx = rng.integers(0, int(ne), int(ne))
        Xb.append(X[idx])
        yb.append(y[idx])

    return Xb, yb, np.asarray(n_list, int)

# From original notebook, cell 35
def _solve_testtype(
    X_list, y_list, n_list, k0, gamma, solver,
    superset, stat=None, solver_kwargs=None
):
    solver = solver.lower()
    kw = {} if solver_kwargs is None else dict(solver_kwargs)
    S0 = sorted(set(superset)-{k0})

    if solver == "cd":
        if stat is None: stat = precompute_stat_test(X_list, y_list, n_list)
        return cd_testtype(stat, k0, gamma=gamma, S_init=S0, superset_init=S0, **kw)

    if solver == "mip":
        obj, B, g = vanilla_testtype(
            X_list, y_list, k0, n_list=n_list, gamma=gamma, S_start=S0, **kw
        )
        return np.flatnonzero(g).tolist(), B, obj

    raise ValueError("solver must be 'cd' or 'mip'.")

# From original notebook, cell 36
def build_supercollection_test(
    X_list, y_list, n_list, k0, superset,
    gamma=(1,1,1,1), M_bar=5, solver="cd",
    random_state=None, solver_kwargs=None
):
    rng = np.random.default_rng(random_state)
    S_list = []

    for _ in range(M_bar):
        Xb, yb, nb = bootstrap_env_data(X_list, y_list, n_list, rng)
        S, _, _ = _solve_testtype(
            Xb, yb, nb, k0, gamma, solver, superset,
            solver_kwargs=solver_kwargs
        )
        S_list.append(tuple(sorted(S)))

    return [list(S) for S in dict.fromkeys(S_list)]

# From original notebook, cell 37
def mip_test_multiple(
    X_D1_list, y_D1_list, n_D1_list,
    X_D2_list, y_D2_list, n_D2_list,
    superset, gamma=(1,1,1,1), alpha=.05,
    M_bar=5, B=300, solver="cd",
    random_state=None, solver_kwargs=None,
    verbose=1, return_details=False
):
    p = X_D1_list[0].shape[1]
    rng = np.random.default_rng(random_state)

    collections = {
        k0: build_supercollection_test(
            X_D1_list, y_D1_list, n_D1_list, k0, superset,
            gamma=gamma, M_bar=M_bar, solver=solver,
            random_state=int(rng.integers(2**32)),
            solver_kwargs=solver_kwargs
        )
        for k0 in range(p)
    }

    stat_D2 = (
        precompute_stat_test(X_D2_list, y_D2_list, n_D2_list)
        if solver.lower() == "cd" else None
    )

    selected, details = [], {}

    for k0 in range(p):
        boot = bootstrap_testtype(
            X_D2_list, y_D2_list, n_D2_list, collections[k0],
            gamma=gamma, B=B, random_state=int(rng.integers(2**32))
        )
        thres = np.quantile(boot, 1-alpha/p).item()

        S, _, T = _solve_testtype(
            X_D2_list, y_D2_list, n_D2_list, k0,
            gamma, solver, superset, stat_D2, solver_kwargs
        )

        if T > thres: selected.append(k0)
        details[k0] = {"T": T, "threshold": thres, "S": S, "S_collection": collections[k0]}

        if verbose:
            print(f"k={k0}: T={T:.6g}, threshold={thres:.6g}, S={S}, M={len(collections[k0])}")

    return (selected, details) if return_details else selected

