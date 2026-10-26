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

RS = np.zeros((0, 7))
n_grid = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
n_trials = args.trials or 200

pbar = tqdm(total=len(n_grid)*n_trials)

for n in n_grid:
    for l in range(n_trials):

# for n in [5000]:
#     for l in range(1):
        pbar.set_postfix(n=n, l=l)

        E_train = 10
        p_a, p_d = np.random.choice([2,3], 2)
        X_list, y_list, n_list, A, true_invariant, D, I, coef, params = generate_hidden_sem_data(
            n=n, E=E_train,
            p_a=p_a, p_s=4, p_d=p_d, k=8, q=2,
            beta=np.ones(4),
            theta_D=np.full(p_d, .1),
            a_scales=[.2+.05*e for e in range(E_train)],
            s_scales=[.2+.05*e for e in range(E_train)],
            d_scales=[.2+.05*e for e in range(E_train)],
            sigma_y=.2,
            gamma_a_scale=.3,
            gamma_s_scale=.3,
            gamma_d_scale=.3,
            b_as_scale=.3
        )
        p = X_list[0].shape[1]
        super_structure, _ = ridge_screening(X_list, y_list, min(p, 100))
        super_structure_10, _ = ridge_screening(X_list, y_list, 10)
        super_structure_15, _ = ridge_screening(X_list, y_list, min(p, 15))

        ## MIP
        t1 = time.time()
        objval, B, g_sol = vanilla_msetype_additive(X_list, y_list, n_list, gamma=300, lam=0.5, s_max=10, mip_focus=3, mip_cuts=3, S_start=None, time_limit = 60, verbose = False)
        t2 = time.time()
        inv_set = np.where(g_sol!=0)[0]
        coef_hat = B
        rs = np.array([
            n,l,
            "MIP-hidden",
            (np.linalg.norm(coef_hat - coef)**2).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished MIP")

        ## EILLS
        X_list_screen = []
        for e in range(len(X_list)):
            X_list_screen.append( X_list[e][:,super_structure_15] )
        t1 = time.time()
        what, g_eills = fit_eills(X_list_screen, y_list, include_intercept=False, gamma=20)
        t2 = time.time()
        if len(g_eills) == 0:
            inv_set = []
        else:
            inv_set = super_structure_15[g_eills] 
        coef_hat = np.zeros(p)
        coef_hat[super_structure_15] = what
        rs = np.array([
            n,l,
            "EILLS-s",
            (np.linalg.norm(coef_hat - coef)**2).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished EILLS")

        # Hidden ICP
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
            n,l,
            "Causal Dantzig",
            (np.linalg.norm(coef_hat - coef)**2).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished Causal Dantzig")

        # BIP-s
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
        inv_set = super_structure_10[np.where(map_z !=0 )[0]]
        coef_hat = np.zeros(p)
        coef_hat[super_structure_10] = ols_result[5][map_idx]
        rs = np.array([
            n,l,
            "BIP-s",
            (np.linalg.norm(coef_hat - coef)**2).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished BIP")

        # negDRO
        t1 = time.time()
        b_final, _ = negDRO(X_list, y_list, gamma = 1000)
        t2 = time.time()
        coef_hat = b_final
        rs = np.array([
            n,l,
            "negDRO",
            (np.linalg.norm(coef_hat - coef)**2).item(),
            np.nan,
            np.nan,
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished negDRO")

        # CoCo
        t1 = time.time()
        res = fit_coco_linear_regularized(
            X_list=X_list,
            y_list=y_list,
            n_list=n_list,
            lam=10.0,
            lr=1e-2,
            max_iter=10000,
            include_intercept=False,
            equal_env_weight=True,
            penalty_sqrt=False,
            verbose=False,
        )
        t2 = time.time()
        coef_hat = res["beta"]
        rs = np.array([
            n,l,
            "CoCo",
            (np.linalg.norm(coef_hat - coef)**2).item(),
            np.nan,
            np.nan,
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished CoCo")

        pbar.update(1)

        print("finished", "n=",n, "l=",l, '\n')

pbar.close()

# Keep figure data and generated outputs in separate files.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
