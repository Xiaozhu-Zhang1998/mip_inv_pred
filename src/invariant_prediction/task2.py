"""Task 2: minimum invariance scores excluding a feature."""
import numpy as np

# From original notebook, cell 17
def vanilla_testtype(
    X_list, y_list, k0, n_list=None, gamma=(1,1,1,1),
    s_max=None, beta_bound=100., S_start=None, B_start=None,
    mip_focus=2, mip_cuts=2, forbidden=None, time_limit=None,
    mip_gap=None, verbose=True
):
    gamma = np.asarray(gamma, float)
    if gamma.shape != (4,) or np.any(gamma < 0):
        raise ValueError("gamma must contain four nonnegative values.")
    gamma1, gamma2, gamma3, gamma4 = gamma

    E = len(X_list)
    if E == 0: raise ValueError("X_list must contain at least one environment.")

    X_list = [np.asarray(X, float) for X in X_list]
    y_list = [np.asarray(y, float).reshape(-1) for y in y_list]
    p = X_list[0].shape[1]
    if not 0 <= k0 < p: raise ValueError("Invalid k0.")

    for e, (X, y) in enumerate(zip(X_list, y_list)):
        if X.ndim != 2 or X.shape != (len(y), p):
            raise ValueError(f"Incompatible X/y dimensions in environment {e}.")

    n_list = np.array([len(y) for y in y_list], float) if n_list is None else np.asarray(n_list, float)
    if len(n_list) != E or np.any(n_list <= 0): raise ValueError("Invalid n_list.")

    w = n_list / n_list.sum()
    wp = np.outer(w, w)

    Sigma = np.array([X.T @ X/n for X, n in zip(X_list, n_list)])
    Xty = np.array([X.T @ y/n for X, y, n in zip(X_list, y_list, n_list)])
    y2 = np.array([y @ y/n for y, n in zip(y_list, n_list)])
    xm = np.array([X.sum(0)/n for X, n in zip(X_list, n_list)])
    ym = np.array([y.sum()/n for y, n in zip(y_list, n_list)])

    XtX_pool = sum(X.T @ X for X in X_list)
    Xty_pool = sum(X.T @ y for X, y in zip(X_list, y_list))

    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model("mip_test_four_term")
    if not verbose: model.Params.OutputFlag = 0
    if time_limit is not None: model.Params.TimeLimit = time_limit
    if mip_gap is not None: model.Params.MIPGap = mip_gap
    model.Params.MIPFocus, model.Params.Cuts = mip_focus, mip_cuts

    g = model.addVars(p, vtype=GRB.BINARY, name="g")
    beta = model.addVars(p, lb=-beta_bound, ub=beta_bound, name="beta")
    beta_e = model.addVars(E, p, lb=-beta_bound, ub=beta_bound, name="beta_e")
    t1, t2, t3, t4 = [model.addVar(lb=0., name=f"t{i}") for i in range(1,5)]

    model.addConstr(g[k0] == 0, name=f"exclude_{k0}")
    if s_max is not None:
        model.addConstr(gp.quicksum(g[j] for j in range(p)) <= s_max, name="sparsity")

    if forbidden is not None:
        for S in forbidden:
            S = set(S)
            model.addConstr(
                gp.quicksum(1-g[j] for j in S) +
                gp.quicksum(g[j] for j in range(p) if j not in S) >= 1
            )

    # Support + environment/pooled OLS
    for j in range(p):
        model.addGenConstrIndicator(g[j], 0, beta[j] == 0, name=f"beta_support_{j}")

        for e in range(E):
            model.addGenConstrIndicator(g[j], 0, beta_e[e,j] == 0, name=f"beta_e_support_{e}_{j}")
            stat_e = gp.quicksum(Sigma[e,j,k]*beta_e[e,k] for k in range(p)) - Xty[e,j]
            model.addGenConstrIndicator(g[j], 1, stat_e == 0, name=f"env_ols_{e}_{j}")

        stat_pool = gp.quicksum(
            w[e] * (gp.quicksum(Sigma[e,j,k]*beta[k] for k in range(p)) - Xty[e,j])
            for e in range(E)
        )
        model.addGenConstrIndicator(g[j], 1, stat_pool == 0, name=f"pooled_ols_{j}")

    # (13a): coefficient mismatch
    q1 = gp.quicksum(
        wp[e,f] * gp.quicksum((beta_e[e,j]-beta_e[f,j])**2 for j in range(p))
        for e in range(E) for f in range(e+1,E)
    )
    model.addQConstr(q1 <= t1, name="term1_coef")

    # (13b): residual variance mismatch
    q2 = gp.QuadExpr()
    for e in range(E):
        for f in range(e+1,E):
            d = (
                y2[e] - y2[f]
                - gp.quicksum(Xty[e,j]*beta_e[e,j] for j in range(p))
                + gp.quicksum(Xty[f,j]*beta_e[f,j] for j in range(p))
            )
            q2 += wp[e,f] * d**2
    model.addQConstr(q2 <= t2, name="term2_var")

    # (13c): residual mean mismatch
    rmean = [
        ym[e] - gp.quicksum(xm[e,j]*beta[j] for j in range(p))
        for e in range(E)
    ]
    q3 = gp.quicksum(
        wp[e,f] * (rmean[e]-rmean[f])**2
        for e in range(E) for f in range(e+1,E)
    )
    model.addQConstr(q3 <= t3, name="term3_mean")

    # (13d): residual normal-equation mismatch for active features
    q4_aux = model.addVars(p, lb=0., name="q4_aux")

    for k in range(p):
        rcorr = [
            Xty[e,k] - gp.quicksum(Sigma[e,k,j]*beta[j] for j in range(p))
            for e in range(E)
        ]
        q4 = gp.quicksum(
            wp[e,f] * (rcorr[e]-rcorr[f])**2
            for e in range(E) for f in range(e+1,E)
        )
        model.addQConstr(q4 <= q4_aux[k], name=f"term4_quad_{k}")
        model.addGenConstrIndicator(g[k], 1, q4_aux[k] <= t4, name=f"term4_active_{k}")

    # Warm start
    if B_start is not None and S_start is None:
        raise ValueError("B_start requires S_start.")

    if S_start is not None:
        S0 = np.array(sorted(set(S_start)), int)
        if np.any(S0 < 0) or np.any(S0 >= p): raise ValueError("Invalid S_start.")
        if k0 in S0: raise ValueError("S_start cannot contain k0.")
        if s_max is not None and len(S0) > s_max: raise ValueError("S_start violates s_max.")

        if B_start is None:
            B_start = np.zeros((E+1,p))
            if len(S0):
                B_start[0,S0] = np.linalg.solve(
                    XtX_pool[np.ix_(S0,S0)], Xty_pool[S0]
                )
                for e in range(E):
                    B_start[e+1,S0] = np.linalg.solve(
                        Sigma[e][np.ix_(S0,S0)], Xty[e,S0]
                    )
        else:
            B_start = np.asarray(B_start, float)
            if B_start.shape != (E+1,p):
                raise ValueError(f"B_start must have shape {(E+1,p)}.")

        S0 = set(S0)
        for j in range(p):
            g[j].Start = float(j in S0)
            beta[j].Start = B_start[0,j]
        for e in range(E):
            for j in range(p):
                beta_e[e,j].Start = B_start[e+1,j]

    model.setObjective(gamma1*t1 + gamma2*t2 + gamma3*t3 + gamma4*t4, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi returned status {model.Status}.")
    if model.SolCount == 0: raise RuntimeError("No feasible solution found.")

    g_sol = np.array([g[j].X > .5 for j in range(p)], int)
    B = np.zeros((E+1,p))
    B[0] = [beta[j].X for j in range(p)]
    for e in range(E): B[e+1] = [beta_e[e,j].X for j in range(p)]

    return model.ObjVal, B, g_sol

# From original notebook, cell 19
def precompute_stat_test(X_list, y_list, n_list=None):
    E = len(X_list)
    n = np.array([len(y) for y in y_list], float) if n_list is None else np.asarray(n_list, float)
    w = n / n.sum()

    y2 = [float(y @ y) for y in y_list]
    Xty = [X.T @ y for X, y in zip(X_list, y_list)]
    XtX = [X.T @ X for X in X_list]

    return {
        "E": E, "n_list": n, "w": w, "w_pair": np.outer(w, w),
        "y2_list": y2, "Xty_list": Xty, "XtX_list": XtX,
        "xmean_list": [X.mean(0) for X in X_list],
        "ymean_list": [float(y.mean()) for y in y_list],
        "Xty_pool": np.sum(Xty, axis=0),
        "XtX_pool": np.sum(XtX, axis=0)
    }

# From original notebook, cell 20
def _add_feature_single(beta, S, Ginv, j, XtX, Xty, tol=1e-12):
    S, p = np.asarray(S, int), len(beta)
    if j in S: raise ValueError(f"j={j} already in S.")

    if len(S) == 0:
        delta = XtX[j, j]
        if delta <= tol: raise np.linalg.LinAlgError(f"Cannot add j={j}.")
        S_new, G_new = np.array([j]), np.array([[1 / delta]])
    else:
        c, d = XtX[S, j], XtX[j, j]
        u, delta = Ginv @ c, d - c @ Ginv @ c
        if delta <= tol: raise np.linalg.LinAlgError(f"Cannot add j={j}.")

        k = len(S)
        G = np.empty((k + 1, k + 1))
        G[:k, :k] = Ginv + np.outer(u, u) / delta
        G[:k, k] = G[k, :k] = -u / delta
        G[k, k] = 1 / delta

        Stmp = np.r_[S, j]
        perm = np.argsort(Stmp)
        S_new, G_new = Stmp[perm], G[np.ix_(perm, perm)]
        G_new = (G_new + G_new.T) / 2

    beta_new = np.zeros(p)
    beta_new[S_new] = G_new @ Xty[S_new]
    return beta_new, S_new, G_new

# From original notebook, cell 21
def add_feature_to_B_ols(B, S, j, Gp, Ge, stat, tol=1e-12):
    E = stat["E"]
    B_new = np.zeros_like(B, float)

    B_new[0], S_new, Gp_new = _add_feature_single(
        B[0], S, Gp, j, stat["XtX_pool"], stat["Xty_pool"], tol
    )

    Ge_new = []
    for e in range(E):
        B_new[e+1], S_e, G_e = _add_feature_single(
            B[e+1], S, Ge[e], j, stat["XtX_list"][e], stat["Xty_list"][e], tol
        )
        if not np.array_equal(S_e, S_new): raise RuntimeError("Support mismatch.")
        Ge_new.append(G_e)

    return B_new, S_new, Gp_new, Ge_new

# From original notebook, cell 22
def _remove_feature_single(beta, S, Ginv, j, Xty, tol=1e-12):
    S, p = np.asarray(S, int), len(beta)
    if j not in S: raise ValueError(f"j={j} not in S.")
    if len(S) == 1: return np.zeros(p), np.array([], int), np.zeros((0, 0))

    pos = np.where(S == j)[0][0]
    keep = np.arange(len(S)) != pos
    d = Ginv[pos, pos]
    if abs(d) <= tol: raise np.linalg.LinAlgError(f"Cannot remove j={j}.")

    G_new = Ginv[np.ix_(keep, keep)] - np.outer(Ginv[keep, pos], Ginv[pos, keep]) / d
    G_new = (G_new + G_new.T) / 2
    S_new = S[keep]

    beta_new = np.zeros(p)
    beta_new[S_new] = G_new @ Xty[S_new]
    return beta_new, S_new, G_new

# From original notebook, cell 23
def remove_feature_from_B_ols(B, S, j, Gp, Ge, stat, tol=1e-12):
    E = stat["E"]
    B_new = np.zeros_like(B, float)

    B_new[0], S_new, Gp_new = _remove_feature_single(
        B[0], S, Gp, j, stat["Xty_pool"], tol
    )

    Ge_new = []
    for e in range(E):
        B_new[e+1], S_e, G_e = _remove_feature_single(
            B[e+1], S, Ge[e], j, stat["Xty_list"][e], tol
        )
        if not np.array_equal(S_e, S_new): raise RuntimeError("Support mismatch.")
        Ge_new.append(G_e)

    return B_new, S_new, Gp_new, Ge_new

# From original notebook, cell 24
def compute_Ginv_given_S(S, stat):
    S, E = np.asarray(S, int), stat["E"]
    if len(S) == 0:
        return np.zeros((0, 0)), [np.zeros((0, 0)) for _ in range(E)]

    Gp = np.linalg.inv(stat["XtX_pool"][np.ix_(S, S)])
    Ge = [
        np.linalg.inv(stat["XtX_list"][e][np.ix_(S, S)])
        for e in range(E)
    ]
    return Gp, Ge

# From original notebook, cell 25
def compute_B_given_S(S, stat):
    S = np.asarray(S, int)
    E, p = stat["E"], stat["XtX_pool"].shape[0]
    B = np.zeros((E + 1, p))
    if len(S) == 0: return B

    Gp, Ge = compute_Ginv_given_S(S, stat)
    B[0, S] = Gp @ stat["Xty_pool"][S]
    for e in range(E):
        B[e+1, S] = Ge[e] @ stat["Xty_list"][e][S]

    return B

# From original notebook, cell 26
def compute_test_obj_given_S(B, S, stat, gamma, return_parts=False):
    gamma = np.asarray(gamma, float)
    if gamma.shape != (4,) or np.any(gamma < 0):
        raise ValueError("gamma must contain four nonnegative values.")
    gamma1, gamma2, gamma3, gamma4 = gamma

    E, n, wp = stat["E"], stat["n_list"], stat["w_pair"]
    y2, Xty, XtX = stat["y2_list"], stat["Xty_list"], stat["XtX_list"]
    xm, ym, beta = stat["xmean_list"], stat["ymean_list"], B[0]
    S = set(map(int, S))

    t1 = sum(
        wp[e,f] * np.sum((B[e+1]-B[f+1])**2)
        for e in range(E) for f in range(e+1,E)
    )

    rvar = np.array([(y2[e]-Xty[e]@B[e+1])/n[e] for e in range(E)])
    t2 = sum(
        wp[e,f] * (rvar[e]-rvar[f])**2
        for e in range(E) for f in range(e+1,E)
    )

    rmean = np.array([ym[e]-xm[e]@beta for e in range(E)])
    t3 = sum(
        wp[e,f] * (rmean[e]-rmean[f])**2
        for e in range(E) for f in range(e+1,E)
    )

    rcorr = np.array([(Xty[e]-XtX[e]@beta)/n[e] for e in range(E)])
    t4_each = sum(
        wp[e,f] * (rcorr[e]-rcorr[f])**2
        for e in range(E) for f in range(e+1,E)
    )
    active = list(S)
    t4 = np.max(t4_each[active]) if active else 0.

    obj = gamma1*t1 + gamma2*t2 + gamma3*t3 + gamma4*t4

    if return_parts:
        return obj, {
            "t1": t1, "t2": t2, "t3": t3, "t4": t4,
            "term1": gamma1*t1, "term2": gamma2*t2,
            "term3": gamma3*t3, "term4": gamma4*t4,
            "total": obj
        }
    return obj

# From original notebook, cell 27
def cd_testtype(
    stat, k0, gamma = (1,1,1,1),
    forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
):
    p = stat["XtX_pool"].shape[0]
    if not 0 <= k0 < p: raise ValueError("Invalid k0.")

    forbid = (set() if forbid is None else set(forbid)) | {k0}
    S = set(S_init) - forbid
    superset = (set(superset_init) - forbid) | S

    S0 = np.array(sorted(S), int)
    B = compute_B_given_S(S0, stat)
    Gp, Ge = compute_Ginv_given_S(S0, stat)
    obj = compute_test_obj_given_S(B, S0, stat, gamma)

    for _ in range(max_iter):

        # Stage 1
        while True:
            improved = False
            for j in sorted(superset):
                S0 = np.array(sorted(S), int)
                out = (
                    remove_feature_from_B_ols(B, S0, j, Gp, Ge, stat, tol)
                    if j in S else
                    add_feature_to_B_ols(B, S0, j, Gp, Ge, stat, tol)
                )
                Bn, Sn, Gpn, Gen = out
                objn = compute_test_obj_given_S(
                    Bn, Sn, stat, gamma
                )
                if objn < obj - tol:
                    S, B, Gp, Ge, obj = set(Sn), Bn, Gpn, Gen, objn
                    improved = True

            if not improved: break

        # Stage 2
        new_feature = False

        for j in range(p):
            if j in forbid: continue
            S0 = np.array(sorted(S), int)
            out = (
                remove_feature_from_B_ols(B, S0, j, Gp, Ge, stat, tol)
                if j in S else
                add_feature_to_B_ols(B, S0, j, Gp, Ge, stat, tol)
            )
            Bn, Sn, Gpn, Gen = out
            objn = compute_test_obj_given_S(
                Bn, Sn, stat, gamma
            )
            if objn < obj - tol:
                S, B, Gp, Ge, obj = set(Sn), Bn, Gpn, Gen, objn

                if j not in superset and j in S:
                    superset.add(j)
                    new_feature = True
                break

        if not new_feature: break

    S = sorted(map(int, S))
    return S, B, compute_test_obj_given_S(
        B, S, stat, gamma
    )

