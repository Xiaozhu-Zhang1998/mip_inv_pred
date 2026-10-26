"""Experiment extracted from summary_pipeline.ipynb; parameters not yet reconciled."""
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

# Notebook cell 104
S_invariant = {
    frozenset([0,1]), frozenset([0,2]), frozenset([0,1,2]), frozenset([0,1,3]),
    frozenset([0,2,3]), frozenset([0,1,2,3])
}

def is_invariant(S):
    return frozenset(S) in S_invariant

# Notebook cell 105
RS = np.zeros([0, 9])

for n_ in [400, 800, 1200, 1600, 2000, 2400, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    for l in range(args.trials or 200):

        X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list, coef = generate_multiple_invariant(
            n_train=300, n_test=n_,
            k=9,
            noise_std=1,
            lbd=1,
            ubd=1,
            perturb_strength=3
        )

        p = X_train_list[0].shape[1]
        true_invariant = [0,2]
        E = len(X_train_list)
        X_list = [np.vstack([X_train_list[e], X_test_list[e]]) for e in range(E)]
        y_list = [np.concatenate([y_train_list[e], y_test_list[e]]) for e in range(E)]
        n_list = [n_train_list[e] + n_test_list[e] for e in range(E)]

        X = np.vstack(X_list)          # shape: (sum(n_list), p)
        y = np.concatenate(y_list)     # shape: (sum(n_list),)
        n_es = n_list                  # e.g. [500, 500, 500]
        n = sum(n_es)
        superset, _ = ridge_screening(X_train_list, y_train_list, 15)
        super_structure_10, _ = ridge_screening(X_train_list, y_train_list, 10)

        
        ## MIP-2
        t1 = time.time()
        inv_set = mip_test_multiple(
            X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list, 
            superset, gamma=(1,1,0,0), alpha=.05,
            M_bar=5, B=300, solver="cd", verbose=0,
        )
        t2 = time.time()
        rs = np.array([
            n_, l, "MIP-test-2", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished MIP-2")

        ## MIP-4
        t1 = time.time()
        inv_set = mip_test_multiple(
            X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list, 
            superset, gamma=(1,1,1,1), alpha=.05,
            M_bar=5, B=300, solver="cd", verbose=0,
        )
        t2 = time.time()
        rs = np.array([
            n_, l, "MIP-test-4", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished MIP-4")

        ### ICP
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
            n_, l, "ICP", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished ICP")

        ### Wasserstein
        wv = DirectWassersteinVariance(X, y, n_es, np.array(n_es) / n)
        _, causes, _, _, _ = WV_screening(wv, alpha=0.05, B=300)
        inv_set = np.where(causes == True)[0]
        rs = np.array([
            n_, l, "Wasserstein", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished Wasserstein")

        ### MIP
        t1 = time.time()
        stat = precompute_stat(X_list, y_list, n_list=None)
        inv_set, B, obj = cd_msetype(
            stat, lam=0.01, gamma1=5, rho=1, forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
        )
        t2 = time.time()
        rs = np.array([
            n_, l, "MIP", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished MIP")

        ### EILLS
        t1 = time.time()
        coef_eills, g_eills = fit_eills(X_list, y_list, include_intercept=False, gamma=20)
        t2 = time.time()
        if len(g_eills) == 0:
            inv_set = []
        else:
            inv_set = g_eills 
        rs = np.array([
            n_, l, "EILLS", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished EILLS")

        ### Causal Dantzig
        X = np.vstack(X_list)      # shape (n_total, p)
        Y = np.concatenate(y_list) # shape (n_total,)
        ExpInd = []
        start = 0
        for i, Xi in enumerate(X_list):
            n_i = Xi.shape[0]
            ExpInd.extend([i]*n_i)
        ExpInd = np.array(ExpInd)  # length n_total
        t1 = time.time()
        hiddenICPobj = fit_hiddenICP(X, Y, ExpInd, alpha=0.05, intercept=False)
        t2 = time.time()
        inv_set = np.where( hiddenICPobj['maximinCoefficients'] != 0 )[0]
        rs = np.array([
            n_, l, "Causal Dantzig", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished Causal Dantzig")

        ### BIP-s
        X_list_screen = []
        for e in range(len(X_list)):
            X_list_screen.append( X_list[e][:,super_structure_10] )
        xs = []
        ys = []
        for i in range(len(X_list_screen)):
            xs.append(torch.tensor(X_list_screen[i]))
            ys.append(torch.tensor(y_list[i]))
        t1 = time.time()
        ols_result, log_posterior, posterior = pi_analytical(
            zs=None,        # None = enumerate all subsets automatically
            xs=xs,          # list of tensors or (E, N, p) tensor
            ys=ys,          # list of tensors or (E, N) tensor
            p_max=20,     # optional: restrict to subsets of size <= p_max
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
        inv_set = super_structure_10[np.where(map_z !=0 )[0]]
        rs = np.array([
            n_, l, "BIP-s", 
            len(inv_set),
            set(inv_set) == set([0]),
            set(inv_set).issubset(set(true_invariant)),
            is_invariant(inv_set),
            len(set(inv_set) - {0,2}),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished BIP-s")

        print("finished n=", n_, "l=", l)
        print(RS)

# Preserve manuscript-independent result file locations.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
