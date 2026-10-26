"""Ridge screening and leave-one-environment-out validation."""
import numpy as np

# From original notebook, cell 53
def ridge_screening(X_list, y_list, k, lam=1.0, intercept=False):
    """
    X_list: list of X^e, each shape (n_e, p)
    y_list: list of y^e
    k: number of features to keep
    lam: ridge regularization
    """

    if intercept:
        k = k-1

    # Stack environments
    X_pool = np.vstack(X_list)
    y_pool = np.concatenate(y_list)
    _, p = X_pool.shape

    # Ridge solution
    XtX = X_pool.T @ X_pool
    Xty = X_pool.T @ y_pool
    beta_hat = np.linalg.solve(XtX + lam * np.eye(p), Xty)

    # Select top-k
    idx_sorted = np.argsort(np.abs(beta_hat))
    S_k = idx_sorted[-k:]

    if intercept:
        S_k = np.insert(S_k, 0, 0)

    return np.sort(S_k), beta_hat

# From original notebook, cell 55
def loeo_mse(X_list, y_list, gamma1, rho, lam, s_max=None, verbose = 0):
    from .task1 import precompute_stat,cd_msetype
    E = len(X_list)
    mse = []
    for e in range(E):
        X_train = [X_list[k] for k in range(E) if k != e]
        y_train = [y_list[k] for k in range(E) if k != e]
        # _, B, _ = vanilla_msetype(
        #     X_train, y_train, gamma1=gamma1, rho=rho,
        #     lam=lam, s_max=s_max, verbose=False
        # )
        stat = precompute_stat(X_train, y_train, n_list=None)
        inv_set, B, obj = cd_msetype(
            stat, lam=0.1, gamma1=5, rho=1, forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
        )
        beta = B[0]
        mse.append(np.mean((y_list[e] - X_list[e] @ beta) ** 2))
        if verbose:
            print("finished", e, "in", E)
    return np.mean(mse)

