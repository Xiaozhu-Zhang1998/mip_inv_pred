"""Task 1 extension for noise interventions with hidden confounding."""
import numpy as np

# From original notebook, cell 40
def vanilla_msetype_additive(
    X_list, y_list, n_list=None, gamma=1., lam=1.,
    w=None, w_pair=None, s_max=None, beta_bound=100.,
    S_start=None, forbidden=None, mip_focus=2, mip_cuts=2,
    time_limit=None, mip_gap=None, verbose=True
):
    E = len(X_list)
    if E == 0: raise ValueError("X_list cannot be empty.")

    X_list = [np.asarray(X, float) for X in X_list]
    y_list = [np.asarray(y, float).reshape(-1) for y in y_list]
    p = X_list[0].shape[1]
    if any(X.shape != (len(y), p) for X, y in zip(X_list, y_list)):
        raise ValueError("Incompatible X/y dimensions.")

    n = np.array([len(y) for y in y_list], float) if n_list is None else np.asarray(n_list, float)
    if len(n) != E or np.any(n <= 0): raise ValueError("Invalid n_list.")

    w = n/n.sum() if w is None else np.asarray(w, float)
    wp = np.outer(w, w) if w_pair is None else np.asarray(w_pair, float)
    if w.shape != (E,) or wp.shape != (E, E): raise ValueError("Invalid weights.")

    Sigma = np.array([X.T@X/ne for X, ne in zip(X_list, n)])
    Xty = np.array([X.T@y/ne for X, y, ne in zip(X_list, y_list, n)])
    y2 = np.array([y@y/ne for y, ne in zip(y_list, n)])

    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model("additive_invariance_miqp")
    if not verbose: model.Params.OutputFlag = 0
    if time_limit is not None: model.Params.TimeLimit = time_limit
    if mip_gap is not None: model.Params.MIPGap = mip_gap
    model.Params.MIPFocus, model.Params.Cuts = mip_focus, mip_cuts

    g = model.addVars(p, vtype=GRB.BINARY, name="g")
    beta = model.addVars(p, lb=-beta_bound, ub=beta_bound, name="beta")

    if s_max is not None:
        model.addConstr(gp.quicksum(g[j] for j in range(p)) <= s_max)

    if forbidden is not None:
        for S in forbidden:
            S = set(S)
            model.addConstr(
                gp.quicksum(1-g[j] for j in S)
                + gp.quicksum(g[j] for j in range(p) if j not in S) >= 1
            )

    for j in range(p):
        model.addGenConstrIndicator(g[j], 0, beta[j] == 0.)

    if S_start is not None:
        S0 = set(S_start)
        for j in range(p): g[j].Start = float(j in S0)

    obj = gp.QuadExpr()

    # weighted prediction error
    for e in range(E):
        obj += w[e] * (
            y2[e]
            - 2*gp.quicksum(Xty[e,j]*beta[j] for j in range(p))
            + gp.quicksum(
                Sigma[e,j,k]*beta[j]*beta[k]
                for j in range(p) for k in range(p)
                if Sigma[e,j,k] != 0
            )
        )

    # pairwise residual-moment discrepancy
    for e in range(E):
        for f in range(e+1, E):
            for j in range(p):
                d = (
                    Xty[e,j] - Xty[f,j]
                    + gp.quicksum(
                        (Sigma[f,j,k]-Sigma[e,j,k])*beta[k]
                        for k in range(p)
                        if Sigma[f,j,k] != Sigma[e,j,k]
                    )
                )
                obj += gamma * wp[e,f] * d*d

    obj += lam * gp.quicksum(g[j] for j in range(p))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] or model.SolCount == 0:
        raise RuntimeError(f"No usable solution. Status={model.Status}")

    beta_sol = np.array([beta[j].X for j in range(p)])
    g_sol = np.array([g[j].X > .5 for j in range(p)], int)

    return model.ObjVal, beta_sol, g_sol

