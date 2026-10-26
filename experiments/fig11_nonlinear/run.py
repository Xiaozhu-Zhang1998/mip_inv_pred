"""Experiment extracted from summary_pipeline.ipynb; parameters not yet reconciled."""
import argparse
from pathlib import Path
from invariant_prediction.experiment_api import *
from invariant_prediction.nonlinear import precompute_stat as precompute_stat_nonlinear

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trials',type=int,default=None,help='Override number of replications (for a quick check)')
parser.add_argument('--seed',type=int,default=None,help='Seed NumPy and Python random generators')
parser.add_argument('--output',default=None,help='Generated CSV path; existing paper CSV is never overwritten')
args=parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)

# Notebook cell 125
RS = np.zeros((0, 6))

for l in range(args.trials or 100):
    for n_ in [100, 150, 200, 250, 300, 350, 400, 450, 500]:
        
        true_invariant, beta_star, X_list, y_list, n_list = generate_synthetic_data_all_intervention_nonlinear(
            n=n_,
            p=6,
            E=5,
            p_act=0.2,
            lb=0.1,
            ub=0.5,
            lb_star=0.5,
            ub_star=1.0,
            sigma_x_min=0.5,
            sigma_x_max=1.0,
            sigma_y=0.5,
            p_star_min=2,
            p_star_max=2
        )
        X = np.vstack(X_list)          # shape: (sum(n_list), p)
        y = np.concatenate(y_list)     # shape: (sum(n_list),)
        n_es = n_list                  # e.g. [500, 500, 500]
        n = sum(n_es)

        df = pd.DataFrame(X, columns=[f"X{j}" for j in range(X.shape[1])])
        df["Y"] = y
        df["E"] = np.repeat(np.arange(1, len(n_list)+1), n_list)
        data_dir=HERE/'data'
        data_dir.mkdir(parents=True,exist_ok=True)
        df.to_csv(data_dir/f"data_{n_}_{l}.csv",index=False)
        pd.DataFrame({'true_invariant':true_invariant}).to_csv(data_dir/f"true_{n_}_{l}.csv",index=False)

        ## MIP
        K = 2
        Z_list, groups = fourier_expand(X_list, K=K)
        t1 = time.time()
        stat = precompute_stat_nonlinear(Z_list, y_list, n_list)
        inv_set, B, obj = cd_msetype_grouped(
            stat, groups, lam=.01, gamma1=1/K , rho=1
        )
        t2 = time.time()
        rs = np.array([
            n_, l, 'MIP', 
            set(inv_set) == set(true_invariant),
            len(list(set(inv_set) & set(true_invariant))),
            t2 - t1,
        ])
        RS = np.vstack([RS, rs])
        print("finished MIP")

        ## Wasserstein
        t1 = time.time()
        wv = DirectWassersteinVariance(X, y, n_es, np.array(n_es) / n)
        _, causes, _, _, _ = WV_screening(wv, alpha=0.05, B=300)
        t2 = time.time()
        inv_set = np.where(causes == True)[0]
        rs = np.array([
            n_, l, "Wasserstein", 
            set(inv_set) == set(true_invariant),
            len(list(set(inv_set) & set(true_invariant))),
            t2 - t1,
        ])
        RS = np.vstack([RS, rs])
        print("finished Wasserstein")

        ## FAIR
        t1 = time.time()
        inv_set, probs, gate_history = fit_fair_nn(X_list, y_list, gamma=36, niters=10000, threshold=.9)
        t2 = time.time()
        rs = np.array([
            n_, l, "FAIR", 
            set(inv_set) == set(true_invariant),
            len(list(set(inv_set) & set(true_invariant))),
            t2 - t1,
        ])
        RS = np.vstack([RS, rs])
        print("finished FAIR")

        print('finished', 'l=', l, 'e=', e)
        print(RS[(RS.shape[0]-3):RS.shape[0],:])

# Preserve manuscript-independent result file locations.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
