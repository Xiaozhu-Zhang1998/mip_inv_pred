"""Baseline implementation; see docs/baselines.md for sources and setup."""
import numpy as np

import numpy as np

def compute_loss(x, y, b, intercept):
    import torch
    """
    Compute the mean squared loss for a given environment.
    """
    if intercept:
        x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)
    preds = x @ b
    return torch.mean((y - preds) ** 2)
    
def weight_solver_closed_form(losses, lambda_reg):
    import torch
    """
    Given environments' losses, solve for the optimal weight using closed-form solution
    \\argmax_{w\\in \\Delta^L} \\ell^T w - \\lambda \\|w\\|_2^2
    it has closed-form solution:
    w_l = max((ell_l - mean(ell)) / (2 * lambda) + 1/L, 0)
    and then normalize w to sum to 1.
    """
    L = len(losses)
    losses_np = losses.detach().numpy()
    mean_loss = np.mean(losses_np)
    w_opt = (losses_np - mean_loss) / (2 * lambda_reg) + 1 / L
    w_opt = np.maximum(w_opt, 0)  # Ensure non-negativity
    if np.sum(w_opt) > 0:
        w_opt /= np.sum(w_opt)  # Normalize to sum to 1
    else:
        w_opt = np.ones(L) / L  # Fallback to uniform distribution if all weights are zero
    return torch.tensor(w_opt, dtype=torch.float32)

def negDRO(x_list, y_list, gamma, intercept=False, 
    lr_b=0.01, early_stop=False, 
    verbose=False, seed=None, num_iter=5000, log_interval=100):
    import torch
    """
    Main algorithm for NegDRO (Negative Distributionally Robust Optimization)
    with ridge penalty and coefficient trajectory logging.

    Args:
        x_list (list of torch.Tensor or np.ndarray): Feature matrices for each environment.
        y_list (list of torch.Tensor or np.ndarray): Target vectors for each environment.
        gamma (float): Regularization parameter controlling the negative weighting.
        intercept (bool): If True, includes an intercept term in the model.
        lr_b (float): Learning rate for parameter b.
        early_stop (bool): If True, enables early stopping based on gradient norm.
        verbose (bool): If True, prints progress updates.
        seed (int, optional): Random seed for reproducibility.
        num_iter (int): Number of iterations for optimization.
        log_interval (int): Interval (in iterations) for logging coefficient trajectories.

    Returns:
        b_final (np.ndarray): Final coefficient vector.
        b_traj (list of np.ndarray): Logged coefficient trajectories during training.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    L = len(x_list)
    _, p = x_list[0].shape

    # Convert all inputs to float tensors
    x_list = [torch.tensor(x, dtype=torch.float32) for x in x_list]
    y_list = [torch.tensor(y, dtype=torch.float32) for y in y_list]

    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize coefficients
    b_dim = p + 1 if intercept else p
    b = torch.nn.Parameter(torch.randn(b_dim) * 0.5) # / torch.sqrt(torch.tensor(b_dim, dtype=torch.float32)))

    # Initialize uniform weights over environments
    w = torch.ones(L) / L
    w.requires_grad = False

    # Optimizer & scheduler
    optimizer_b = torch.optim.Adam([b], lr=lr_b)
    scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=1000, gamma=0.5)

    # Scheduler for lambda_reg
    lam_schedule_c=1.0           # c in c * M_est / sqrt(t + t0)
    lam_schedule_t0=100          # t0 to avoid overly large weights early on
    M_est=1.0                     # rough smoothness scale; can be tuned or estimated

    # Trackers
    min_grad_norm = float("inf")
    b_best = b.detach().clone()
    converged = False
    b_traj = []  # store coefficient trajectory

    # --------------------------------------------------------------------------
    # Optimization loop
    # --------------------------------------------------------------------------
    for i in range(num_iter):
        optimizer_b.zero_grad()

        # Compute per-environment losses
        losses = [compute_loss(x_list[e], y_list[e], b, intercept) for e in range(L)]
        losses_tensor = torch.stack(losses)
    
        lambda_reg_t = lam_schedule_c * M_est / np.sqrt(i + lam_schedule_t0)

        # Solve for environment weights in closed form
        w = weight_solver_closed_form(losses_tensor.detach(), lambda_reg_t)
        w.requires_grad = False

        # Compute total objective
        total_loss = torch.sum((w - gamma / (1 + gamma * L)) * losses_tensor) \
                     - lambda_reg_t * torch.sum(w ** 2)

        # Backpropagation
        total_loss.backward()
        grad_norm = torch.norm(b.grad).item()

        # Track best (lowest-gradient) b
        if grad_norm < min_grad_norm:
            min_grad_norm = grad_norm
            b_best = b.detach().clone()

        # Early stopping (optional)
        if early_stop and grad_norm < 1e-4:
            converged = True
            if verbose:
                print(f"[Early Stop] Iter {i:4d} | Loss={total_loss.item():.4f} | GradNorm={grad_norm:.5e}")
            break

        # Safety stop if gradient explodes
        if grad_norm > 5e3:
            print("⚠️ Gradient explosion detected. Stopping early.")
            break

        # Update parameters
        optimizer_b.step()
        scheduler_b.step()

        # Log coefficient trajectory
        if (i % log_interval == 0) or (i == num_iter - 1):
            b_traj.append(b.detach().cpu().numpy().flatten())
            if verbose:
                print(f"Iter {i:4d} | Loss={total_loss.item():.4f} | GradNorm={grad_norm:.5e}")

    # --------------------------------------------------------------------------
    # Finalize output
    # --------------------------------------------------------------------------
    if not converged:
        b = b_best

    b_final = b.detach().cpu().numpy().flatten()
    return b_final, b_traj

