"""Task 1: invariant prediction via MIQP and coordinate descent."""
import numpy as np

def vanilla_msetype(
    X_list, y_list, n_list=None, gamma1=1.0, rho=1.0, lam=1.0,
    s_max=None, beta_bound=100.0, S_start=None, B_start=None,
    mip_focus=2, mip_cuts=2, forbidden=None, time_limit=None,
    mip_gap=None, verbose=True
):
    E = len(X_list)
    if E == 0: raise ValueError("X_list must contain at least one environment.")

    X_list = [np.asarray(X, float) for X in X_list]
    y_list = [np.asarray(y, float).reshape(-1) for y in y_list]
    p = X_list[0].shape[1]

    for e, (X, y) in enumerate(zip(X_list, y_list)):
        if X.ndim != 2 or X.shape[1] != p or X.shape[0] != len(y):
            raise ValueError(f"Incompatible X/y dimensions in environment {e}.")

    n_list = np.array([len(y) for y in y_list], float) if n_list is None else np.asarray(n_list, float)
    if len(n_list) != E or np.any(n_list <= 0): raise ValueError("Invalid n_list.")

    w = n_list / n_list.sum()
    w_pair = np.outer(w, w)

    Sigma = np.array([X.T @ X / n for X, n in zip(X_list, n_list)])
    Xty = np.array([X.T @ y / n for X, y, n in zip(X_list, y_list, n_list)])
    y2 = np.array([y @ y / n for y, n in zip(y_list, n_list)])

    XtX_pool = sum(X.T @ X for X in X_list)
    Xty_pool = sum(X.T @ y for X, y in zip(X_list, y_list))

    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model("finite_sample_invariance_indicator_miqp")
    if not verbose: model.Params.OutputFlag = 0
    if time_limit is not None: model.Params.TimeLimit = time_limit
    if mip_gap is not None: model.Params.MIPGap = mip_gap
    model.Params.MIPFocus, model.Params.Cuts = mip_focus, mip_cuts

    g = model.addVars(p, vtype=GRB.BINARY, name="g")
    beta = model.addVars(p, lb=-beta_bound, ub=beta_bound, name="beta")
    beta_e = model.addVars(E, p, lb=-beta_bound, ub=beta_bound, name="beta_e")

    if s_max is not None:
        model.addConstr(gp.quicksum(g[j] for j in range(p)) <= s_max, name="sparsity")

    if forbidden is not None:
        for S in forbidden:
            S = set(S)
            model.addConstr(
                gp.quicksum(1 - g[j] for j in S)
                + gp.quicksum(g[j] for j in range(p) if j not in S) >= 1
            )

    for j in range(p):
        model.addGenConstrIndicator(g[j], 0, beta[j] == 0, name=f"beta_support_{j}")
        for e in range(E):
            model.addGenConstrIndicator(g[j], 0, beta_e[e, j] == 0, name=f"beta_e_support_{e}_{j}")
            stationarity = gp.quicksum(Sigma[e, j, k] * beta_e[e, k] for k in range(p)) - Xty[e, j]
            model.addGenConstrIndicator(g[j], 1, stationarity == 0, name=f"ols_stationarity_{e}_{j}")

    if B_start is not None and S_start is None:
        raise ValueError("B_start requires S_start.")

    if S_start is not None:
        S0 = np.array(sorted(set(S_start)), int)
        if np.any(S0 < 0) or np.any(S0 >= p): raise ValueError("S_start contains invalid indices.")
        if s_max is not None and len(S0) > s_max: raise ValueError("S_start violates s_max.")

        if B_start is None:
            B_start = np.zeros((E + 1, p))
            if len(S0):
                B_start[0, S0] = np.linalg.solve(XtX_pool[np.ix_(S0, S0)], Xty_pool[S0])
                for e in range(E):
                    B_start[e + 1, S0] = np.linalg.solve(
                        Sigma[e][np.ix_(S0, S0)], Xty[e, S0]
                    )
        else:
            B_start = np.asarray(B_start, float)
            if B_start.shape != (E + 1, p):
                raise ValueError(f"B_start must have shape {(E + 1, p)}.")

        S0 = set(S0)
        for j in range(p):
            g[j].Start = float(j in S0)
            beta[j].Start = B_start[0, j]
        for e in range(E):
            for j in range(p):
                beta_e[e, j].Start = B_start[e + 1, j]

    obj = gp.QuadExpr()

    for e in range(E):
        obj += w[e] * y2[e]
        obj += gp.quicksum(-2 * w[e] * Xty[e, j] * beta[j] for j in range(p))
        obj += gp.quicksum(
            w[e] * Sigma[e, j, k] * beta[j] * beta[k]
            for j in range(p) for k in range(p) if Sigma[e, j, k] != 0
        )

    for e in range(E):
        for f in range(e + 1, E):
            wef = w_pair[e, f]

            obj += gamma1 * wef * gp.quicksum(
                (beta_e[e, j] - beta_e[f, j]) ** 2 for j in range(p)
            )

            var_diff = (
                y2[e] - y2[f]
                - gp.quicksum(Xty[e, j] * beta_e[e, j] for j in range(p))
                + gp.quicksum(Xty[f, j] * beta_e[f, j] for j in range(p))
            )
            obj += gamma1 * rho * wef * var_diff ** 2

    obj += lam * gp.quicksum(g[j] for j in range(p))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi did not return a usable solution. Status = {model.Status}")
    if model.SolCount == 0: raise RuntimeError("Gurobi did not find a feasible solution.")

    g_sol = np.array([g[j].X > 0.5 for j in range(p)], int)
    B = np.zeros((E + 1, p))
    B[0] = [beta[j].X for j in range(p)]
    for e in range(E): B[e + 1] = [beta_e[e, j].X for j in range(p)]

    return model.ObjVal, B, g_sol

def precompute_stat(X_list, y_list, n_list=None):
    E = len(X_list)
    n_list = np.array([len(y) for y in y_list], float) if n_list is None else np.asarray(n_list, float)
    w = n_list / n_list.sum()
    y2 = [float(y @ y) for y in y_list]
    Xty = [X.T @ y for X, y in zip(X_list, y_list)]
    XtX = [X.T @ X for X in X_list]

    return {
        "E": E, "n_list": n_list, "w": w, "w_pair": np.outer(w, w),
        "y2_list": y2, "Xty_list": Xty, "XtX_list": XtX,
        "Xty_pool": np.sum(Xty, axis=0), "XtX_pool": np.sum(XtX, axis=0)
    }

def _add_feature_single(beta, S, Ginv, j, XtX, Xty, tol=1e-12):
    S, p = np.asarray(S, int), len(beta)
    if j in S: raise ValueError(f"j={j} is already in S.")

    if len(S) == 0:
        delta = XtX[j, j]
        if delta <= tol: raise np.linalg.LinAlgError(f"Cannot add j={j}: Schur complement too small.")
        S_new, Ginv_new = np.array([j]), np.array([[1 / delta]])
    else:
        c, d = XtX[S, j], XtX[j, j]
        u, delta = Ginv @ c, d - c @ Ginv @ c
        if delta <= tol: raise np.linalg.LinAlgError(f"Cannot add j={j}: Schur complement too small ({delta}).")
        k = len(S)
        Gtmp = np.empty((k + 1, k + 1))
        Gtmp[:k, :k] = Ginv + np.outer(u, u) / delta
        Gtmp[:k, k] = Gtmp[k, :k] = -u / delta
        Gtmp[k, k] = 1 / delta
        Stmp = np.r_[S, j]
        perm = np.argsort(Stmp)
        S_new, Ginv_new = Stmp[perm], Gtmp[np.ix_(perm, perm)]
        Ginv_new = (Ginv_new + Ginv_new.T) / 2

    beta_new = np.zeros(p)
    beta_new[S_new] = Ginv_new @ Xty[S_new]
    return beta_new, S_new, Ginv_new

def add_feature_to_B_ols(B, S, j, Ginv_pool, Ginv_list, stat, tol=1e-12):
    E = stat["E"]
    B_new = np.zeros_like(B, dtype=float)
    B_new[0], S_new, Ginv_pool_new = _add_feature_single(
        B[0], S, Ginv_pool, j, stat["XtX_pool"], stat["Xty_pool"], tol
    )
    Ginv_list_new = []
    for e in range(E):
        B_new[e + 1], S_e, G_e = _add_feature_single(
            B[e + 1], S, Ginv_list[e], j, stat["XtX_list"][e], stat["Xty_list"][e], tol
        )
        if not np.array_equal(S_e, S_new): raise RuntimeError("Support mismatch across environments.")
        Ginv_list_new.append(G_e)

    return B_new, S_new, Ginv_pool_new, Ginv_list_new

def _remove_feature_single(beta, S, Ginv, j, Xty, tol=1e-12):
    S, p = np.asarray(S, int), len(beta)
    if j not in S: raise ValueError(f"j={j} is not in S.")
    if len(S) == 1:
        return np.zeros(p), np.array([], int), np.zeros((0, 0))
    pos = np.where(S == j)[0][0]
    keep = np.arange(len(S)) != pos
    d = Ginv[pos, pos]
    if abs(d) <= tol: raise np.linalg.LinAlgError(f"Cannot remove j={j}: pivot too small.")
    col, row = Ginv[keep, pos], Ginv[pos, keep]
    Ginv_new = Ginv[np.ix_(keep, keep)] - np.outer(col, row) / d
    Ginv_new = (Ginv_new + Ginv_new.T) / 2
    S_new = S[keep]
    beta_new = np.zeros(p)
    beta_new[S_new] = Ginv_new @ Xty[S_new]
    return beta_new, S_new, Ginv_new

def remove_feature_from_B_ols(B, S, j, Ginv_pool, Ginv_list, stat, tol=1e-12):
    E = stat["E"]
    B_new = np.zeros_like(B, dtype=float)
    B_new[0], S_new, Ginv_pool_new = _remove_feature_single(
        B[0], S, Ginv_pool, j, stat["Xty_pool"], tol
    )
    Ginv_list_new = []
    for e in range(E):
        B_new[e + 1], S_e, G_e = _remove_feature_single(
            B[e + 1], S, Ginv_list[e], j, stat["Xty_list"][e], tol
        )
        if not np.array_equal(S_e, S_new): raise RuntimeError("Support mismatch across environments.")
        Ginv_list_new.append(G_e)
        
    return B_new, S_new, Ginv_pool_new, Ginv_list_new

def compute_Ginv_given_S(S, stat):
    S, E = np.asarray(S, int), stat["E"]
    if len(S) == 0:
        return np.zeros((0, 0)), [np.zeros((0, 0)) for _ in range(E)]
    Ginv_pool = np.linalg.inv(stat["XtX_pool"][np.ix_(S, S)])
    Ginv_list = [np.linalg.inv(stat["XtX_list"][e][np.ix_(S, S)]) for e in range(E)]
    return Ginv_pool, Ginv_list

def compute_B_given_S(S, stat):
    S = np.asarray(S, int)
    E, p = stat["E"], stat["XtX_pool"].shape[0]
    B = np.zeros((E + 1, p))
    if len(S) == 0: return B
    Ginv_pool, Ginv_list = compute_Ginv_given_S(S, stat)
    B[0, S] = Ginv_pool @ stat["Xty_pool"][S]
    for e in range(E): B[e + 1, S] = Ginv_list[e] @ stat["Xty_list"][e][S]
    return B

def compute_mse_obj_given_S(B, S, stat, gamma1=1.0, rho=1.0, lam=0.0, return_parts=False):
    E, n, w, w_pair = stat["E"], stat["n_list"], stat["w"], stat["w_pair"]
    y2, Xty, XtX = stat["y2_list"], stat["Xty_list"], stat["XtX_list"]
    beta = B[0]
    pred = sum(
        w[e] * (y2[e] - 2 * Xty[e] @ beta + beta @ XtX[e] @ beta) / n[e]
        for e in range(E)
    )
    resid = [(y2[e] - Xty[e] @ B[e + 1]) / n[e] for e in range(E)]
    coef_diff, var_diff = 0.0, 0.0
    for e in range(E):
        for f in range(e + 1, E):
            d = B[e + 1] - B[f + 1]
            coef_diff += w_pair[e, f] * (d @ d)
            var_diff += w_pair[e, f] * (resid[e] - resid[f]) ** 2
    sparsity = lam * len(S)
    obj = pred + gamma1 * (coef_diff + rho * var_diff) + sparsity
    if return_parts:
        return obj, {
            "prediction": pred,
            "coef_diff_raw": coef_diff,
            "coef_diff": gamma1 * coef_diff,
            "var_diff_raw": var_diff,
            "var_diff": gamma1 * rho * var_diff,
            "sparsity": sparsity,
            "total": obj
        }
    return obj

def cd_msetype(
    stat, lam, gamma1, rho, forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
):
    p = stat["XtX_pool"].shape[0]
    forbid = set() if forbid is None else set(forbid)
    S = set(S_init) - forbid
    superset = (set(superset_init) - forbid) | S
    S0 = np.array(sorted(S), int)
    B = compute_B_given_S(S0, stat)
    Ginv_pool, Ginv_list = compute_Ginv_given_S(S0, stat)
    obj = compute_mse_obj_given_S(B, S0, stat, gamma1, rho, lam)

    for _ in range(max_iter):

        # Stage 1: converge within active set
        while True:
            improved = False
            for j in sorted(superset):
                S_curr = np.array(sorted(S), int)
                if j in S:
                    out = remove_feature_from_B_ols(B, S_curr, j, Ginv_pool, Ginv_list, stat, tol)
                else:
                    if j in forbid: continue
                    out = add_feature_to_B_ols(B, S_curr, j, Ginv_pool, Ginv_list, stat, tol)
                B_new, S_new, Gp_new, Ge_new = out
                obj_new = compute_mse_obj_given_S(B_new, S_new, stat, gamma1, rho, lam)
                if obj_new < obj - tol:
                    S, B, Ginv_pool, Ginv_list, obj = set(S_new), B_new, Gp_new, Ge_new, obj_new
                    improved = True
            if not improved: break

        # Stage 2: first improving global move
        new_feature = False
        for j in range(p):
            S_curr = np.array(sorted(S), int)
            if j in S:
                out = remove_feature_from_B_ols(B, S_curr, j, Ginv_pool, Ginv_list, stat, tol)
            else:
                if j in forbid: continue
                out = add_feature_to_B_ols(B, S_curr, j, Ginv_pool, Ginv_list, stat, tol)
            B_new, S_new, Gp_new, Ge_new = out
            obj_new = compute_mse_obj_given_S(B_new, S_new, stat, gamma1, rho, lam)
            if obj_new < obj - tol:
                S, B, Ginv_pool, Ginv_list, obj = set(S_new), B_new, Gp_new, Ge_new, obj_new
                if j not in superset and j in S:
                    superset.add(j)
                    new_feature = True
                break
        if not new_feature: break

    S = sorted(map(int, S))
    return S, B, compute_mse_obj_given_S(B, S, stat, gamma1, rho, lam)

