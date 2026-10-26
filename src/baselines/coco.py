"""Baseline implementation; see docs/baselines.md for sources and setup."""
import numpy as np

import numpy as np

def fit_coco_linear_regularized(
    X_list,
    y_list,
    n_list=None,
    lam=1.0,
    lr=1e-2,
    max_iter=10000,
    include_intercept=True,
    equal_env_weight=True,
    penalty_sqrt=False,
    l2=0.0,
    dtype=None,
    device=None,
    seed=None,
    verbose=False,
    print_every=1000,
):
    """
    Fit regularized linear CoCo:

        risk(theta) + lam * coco_penalty(theta)

    where

        risk(theta)
            = average_e 1/(2 n_e) ||y_e - X_e theta||^2,

        coco_penalty(theta)
            = average_e [g_{e,0}^2 + sum_j (beta_j g_{e,j})^2],

    and

        g_e = grad_theta R_e(theta).

    Parameters
    ----------
    X_list : list of arrays
        X_list[e] has shape (n_e, p).
    y_list : list of arrays
        y_list[e] has shape (n_e,).
    n_list : list of int, optional
        Sample sizes. If None, inferred from X_list.
    lam : float
        CoCo regularization strength.
    lr : float
        Adam learning rate.
    max_iter : int
        Number of optimization iterations.
    include_intercept : bool
        If True, include an intercept.
    equal_env_weight : bool
        If True, each environment receives equal weight.
        If False, environments are weighted by sample size.
    penalty_sqrt : bool
        If True, use sqrt(coco_penalty).
        If False, use coco_penalty directly.
    l2 : float
        Optional ridge penalty on slopes only.
    dtype : torch dtype
    device : torch device
    seed : int or None
    verbose : bool
    print_every : int

    Returns
    -------
    result : dict
        Contains beta, intercept, theta, objective, risk, coco_penalty.
    """

    import torch
    if dtype is None: dtype=torch.float64
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device is None:
        device = torch.device("cpu")

    E = len(X_list)

    X_tensors = []
    y_tensors = []

    for X, y in zip(X_list, y_list):
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device).reshape(-1)

        if include_intercept:
            ones = torch.ones((X.shape[0], 1), dtype=dtype, device=device)
            X = torch.cat([ones, X], dim=1)

        X_tensors.append(X)
        y_tensors.append(y)

    if n_list is None:
        n_list = [X.shape[0] for X in X_tensors]

    if equal_env_weight:
        env_weights = torch.ones(E, dtype=dtype, device=device) / E
    else:
        n_total = float(sum(n_list))
        env_weights = torch.tensor(
            [n_e / n_total for n_e in n_list],
            dtype=dtype,
            device=device,
        )

    p_aug = X_tensors[0].shape[1]

    # Initialize at pooled OLS when possible. This is usually more stable than random init.
    X_pool = torch.cat(X_tensors, dim=0)
    y_pool = torch.cat(y_tensors, dim=0)

    try:
        theta_init = torch.linalg.solve(
            X_pool.T @ X_pool + 1e-6 * torch.eye(p_aug, dtype=dtype, device=device),
            X_pool.T @ y_pool,
        )
    except RuntimeError:
        theta_init = torch.zeros(p_aug, dtype=dtype, device=device)

    theta = torch.nn.Parameter(theta_init.clone())

    optimizer = torch.optim.Adam([theta], lr=lr)

    history = []

    for it in range(max_iter):
        optimizer.zero_grad()

        risk = 0.0
        coco_penalty = 0.0

        for w_e, X_e, y_e in zip(env_weights, X_tensors, y_tensors):
            n_e = X_e.shape[0]

            resid_e = X_e @ theta - y_e

            # Environment-specific squared-loss risk:
            # R_e(theta) = 1/(2n_e) ||X_e theta - y_e||^2
            risk_e = 0.5 * torch.mean(resid_e ** 2)

            # Analytic gradient of R_e(theta):
            # g_e = X_e^T (X_e theta - y_e) / n_e
            grad_e = X_e.T @ resid_e / n_e

            if include_intercept:
                # Penalize intercept score directly.
                intercept_score = grad_e[0] ** 2

                # Penalize slope scores multiplied by slope coefficients.
                slope_score = torch.sum((theta[1:] * grad_e[1:]) ** 2)

                coco_e = intercept_score + slope_score
            else:
                coco_e = torch.sum((theta * grad_e) ** 2)

            risk = risk + w_e * risk_e
            coco_penalty = coco_penalty + w_e * coco_e

        if penalty_sqrt:
            coco_term = torch.sqrt(coco_penalty + 1e-12)
        else:
            coco_term = coco_penalty

        if include_intercept:
            ridge_term = l2 * torch.sum(theta[1:] ** 2)
        else:
            ridge_term = l2 * torch.sum(theta ** 2)

        objective = risk + lam * coco_term + ridge_term

        objective.backward()
        optimizer.step()

        history.append(
            {
                "objective": float(objective.detach().cpu()),
                "risk": float(risk.detach().cpu()),
                "coco_penalty": float(coco_penalty.detach().cpu()),
            }
        )

        if verbose and it % print_every == 0:
            print(
                f"iter={it:5d}, "
                f"obj={history[-1]['objective']:.6g}, "
                f"risk={history[-1]['risk']:.6g}, "
                f"coco={history[-1]['coco_penalty']:.6g}"
            )

    theta_hat = theta.detach().cpu().numpy()

    if include_intercept:
        intercept = theta_hat[0]
        beta = theta_hat[1:]
    else:
        intercept = 0.0
        beta = theta_hat

    return {
        "beta": beta,
        "intercept": intercept,
        "theta": theta_hat,
        "objective": history[-1]["objective"],
        "risk": history[-1]["risk"],
        "coco_penalty": history[-1]["coco_penalty"],
        "history": history,
    }

