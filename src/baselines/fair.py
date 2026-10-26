"""FAIR adapter; download the upstream FAIR package separately."""
from copy import deepcopy

# From original notebook, cell 73
from copy import deepcopy

def fit_fair_nn(X_list, y_list, gamma=36, niters=10000,
                threshold=.9, anneal_rate=.96, seed=0):
    try:
        from FAIR.methods.fair_algo import FairMLP,FairGumbelAlgo
        from FAIR.utils import aos_default_hyper_params
    except ImportError as exc:
        raise ImportError("FAIR is optional; see docs/baselines.md for installation.") from exc
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    E, p = len(X_list), X_list[0].shape[1]
    xs = [np.asarray(X, dtype=np.float32) for X in X_list]
    ys = [np.asarray(y, dtype=np.float32).reshape(-1, 1) for y in y_list]
    hp = deepcopy(aos_default_hyper_params)
    hp.update(
        offset=-1, niters=niters,
        init_temp=5., final_temp=.1,
        anneal_iter=100, anneal_rate=anneal_rate
    )
    model = FairMLP(E, p, depth_g=1, width_g=128, depth_f=2, width_f=196)
    loss = lambda yhat, y: .5 * torch.mean((yhat - y)**2)
    algo = FairGumbelAlgo(E, p, model, gamma, loss, hp)
    result = algo.run_gumbel(
        (xs, ys), eval_iter=max(1, niters//10),
        gate_samples=20, log=False
    )
    probs = np.asarray(result["gate_rec"][-1])
    support = np.where(probs > threshold)[0]
    return support, probs, np.asarray(result["gate_rec"])

