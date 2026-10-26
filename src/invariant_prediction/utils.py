"""Ridge screening and leave-one-environment-out validation."""
import numpy as np

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

def loeo_mse(X_list, y_list, gamma1, rho, lam, s_max=None, verbose = 0):
    from .task1 import precompute_stat,cd_msetype
    E = len(X_list)
    mse = []
    for e in range(E):
        X_train = [X_list[k] for k in range(E) if k != e]
        y_train = [y_list[k] for k in range(E) if k != e]
        stat = precompute_stat(X_train, y_train, n_list=None)
        inv_set,B,obj=cd_msetype(stat,lam=lam,gamma1=gamma1,rho=rho,S_init=())
        beta = B[0]
        mse.append(np.mean((y_list[e] - X_list[e] @ beta) ** 2))
        if verbose:
            print("finished", e, "in", E)
    return np.mean(mse)

