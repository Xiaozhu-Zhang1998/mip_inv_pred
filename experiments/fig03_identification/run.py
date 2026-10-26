"""Figure experiment; settings may require further reproducibility checks."""
import argparse
from pathlib import Path
from invariant_prediction.experiment_api import *
import torch

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trials',type=int,default=None,help='Override number of replications (for a quick check)')
parser.add_argument('--seed',type=int,default=None,help='Seed NumPy and Python random generators')
parser.add_argument('--output',default=None,help='Generated CSV path; existing paper CSV is never overwritten')
args=parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)

from tqdm import tqdm
RS = np.zeros([0, 7])

n_trail = args.trials or 5000
pbar = tqdm(total = n_trail)

for l in range(n_trail):
    pbar.set_postfix(l=l)
    X_list, y_list, n_list = generate_X1_to_Y_to_X2_example(
        n_list=(500, 500),
        k_independent=8,
        gamma=1.0,
        tau2=1.0,
        sigma_eps2=1.0,
        a_list=(1.0, 0.5),
        v_list=(2.0, 1.5),
        seed=None,
        return_params=False,
    )
    E = 2
    p = X_list[0].shape[1]
    true_invariant = [0]
    coef = np.zeros(p)
    coef[true_invariant]=1

    ## MIP
    t1 = time.time()
    objval, B, g_sol = vanilla_msetype(X_list, y_list, n_list, gamma1=50, rho=1, lam=0.5, s_max=10, mip_focus=3, mip_cuts=3, S_start=None, time_limit = 60, verbose=0)
    t2 = time.time()
    inv_set = np.where(g_sol!=0)[0]
    coef_hat = B[0]
    rs = np.array([
        l,
        "MIP",
        (np.linalg.norm(coef_hat - coef)**2).item(),
        set(inv_set).issuperset(set(true_invariant)),
        set(inv_set) == set(true_invariant),
        set([1]).issubset(set(inv_set)),
        t2 - t1
    ])
    RS = np.vstack([RS, rs])

    ## EILLS
    t1 = time.time()
    what, g_eills = fit_eills(X_list, y_list, include_intercept=False, gamma=50)
    t2 = time.time()
    if len(g_eills) == 0:
        inv_set = []
    else:
        inv_set = g_eills 
    coef_hat = what
    rs = np.array([
        l,
        "EILLS",
        (np.linalg.norm(coef_hat - coef)**2).item(),
        set(inv_set).issuperset(set(true_invariant)),
        set(inv_set) == set(true_invariant),
        set([1]).issubset(set(inv_set)),
        t2 - t1
    ])
    RS = np.vstack([RS, rs])

    ## ICP
    data = []
    for i in range(len(X_list)):
        data.append( np.column_stack([X_list[i], y_list[i]]) )
    target = X_list[0].shape[1]
    t1 = time.time()
    result = icp.fit(data, target, alpha=0.05, precompute=True, verbose=False, color=False)
    t2 = time.time()
    if len(result.accepted_sets) == 0:
        intersection = []
    else:
        intersection = reduce(lambda a, b: a & b, result.accepted_sets)
    inv_set = list(intersection)
    rs = np.array([
        l,
        "ICP",
        np.nan,
        set(inv_set).issuperset(set(true_invariant)),
        set(inv_set) == set(true_invariant),
        set([1]).issubset(set(inv_set)),
        t2 - t1
    ])
    RS = np.vstack([RS, rs])
        

    ## Hidden ICP
    X = np.vstack(X_list)      # shape (n_total, p)
    Y = np.concatenate(y_list) # shape (n_total,)
    ExpInd = []
    start = 0
    for i, Xi in enumerate(X_list):
        n_i = Xi.shape[0]
        ExpInd.extend([i]*n_i)
    ExpInd = np.array(ExpInd)  # length n_total
    t1 = time.time()
    hiddenICPobj = fit_hiddenICP(X, Y, ExpInd, alpha=0.1, intercept=False)
    t2 = time.time()
    inv_set = np.where( hiddenICPobj['maximinCoefficients'] != 0 )[0]
    coef_hat = hiddenICPobj['maximinCoefficients']
    rs = np.array([
        l,
        "Causal Dantzig",
        (np.linalg.norm(coef_hat - coef)**2).item(),
        set(inv_set).issuperset(set(true_invariant)),
        set(inv_set) == set(true_invariant),
        set([1]).issubset(set(inv_set)),
        t2 - t1
    ])
    RS = np.vstack([RS, rs])

    ## BIP-s
    xs = []
    ys = []
    for i in range(len(X_list)):
        xs.append(torch.tensor(X_list[i]))
        ys.append(torch.tensor(y_list[i]))
    t1 = time.time()
    ols_result, log_posterior, posterior = pi_analytical(
        zs=None,        # None = enumerate all subsets automatically
        xs=xs,          # list of tensors or (E, N, p) tensor
        ys=ys,          # list of tensors or (E, N) tensor
        p_max=10,     # optional: restrict to subsets of size <= p_max
        epsilon=1e-4,   # ridge jitter for OLS stability
        use_vectorize=True,         # True = faster batched computation
        include_intercept=False,     # add intercept column to X
        keep_intercept=False         # always include intercept in z
    )
    t2 = time.time()
    p_ = xs[0].shape[-1]
    zs = get_all_zs(p_, p_max=10)   # all candidate z's in the same order as posterior
    map_idx = posterior.argmax()
    map_z = zs[map_idx]  
    inv_set = np.where(map_z !=0 )[0]
    coef_hat = np.zeros(p)
    coef_hat = ols_result[5][map_idx]
    rs = np.array([
        l,
        "BIP",
        (np.linalg.norm(coef_hat - coef)**2).item(),
        set(inv_set).issuperset(set(true_invariant)),
        set(inv_set) == set(true_invariant),
        set([1]).issubset(set(inv_set)),
        t2 - t1
    ])
    RS = np.vstack([RS, rs])

    pbar.update(1)

pbar.close()

# Keep figure data and generated outputs in separate files.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
