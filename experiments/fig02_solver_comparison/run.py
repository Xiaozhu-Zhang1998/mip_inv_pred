"""Experiment extracted from summary_pipeline.ipynb; parameters not yet reconciled."""
import argparse
from pathlib import Path
from invariant_prediction.experiment_api import *

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trials',type=int,default=None,help='Override number of replications (for a quick check)')
parser.add_argument('--seed',type=int,default=None,help='Seed NumPy and Python random generators')
parser.add_argument('--output',default=None,help='Generated CSV path; existing paper CSV is never overwritten')
args=parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)

# Notebook cell 107
from tqdm import tqdm

RS = np.zeros((0, 8))
p_grid = [10, 20, 30, 40, 50, 60]
n_trials = args.trials or 100

pbar = tqdm(total=len(p_grid)*n_trials)

for p in p_grid:
    for l in range(n_trials):
        pbar.set_postfix(p=p, l=l)
        true_invariant, coef, X_list, y_list, n_list = generate_synthetic_data_soft_intervention(
            n=200,
            p=p,
            E=3,
            p_act=np.random.choice([0.1, 0.2]),
            lb=0.1, ub=0.1,
            lb_star=0.5, ub_star=1,
            sigma_x_min=0.5,
            sigma_x_max=1.0,
            sigma_y=0.5,
            p_star_min=4,
            p_star_max=5,
            p_resample = 0.05
        )

        # MIP-cd
        t1 = time.time()
        stat = precompute_stat(X_list, y_list, n_list=None)
        inv_set, B, obj = cd_msetype(
            stat, lam=0.01, gamma1=5, rho=1, forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
        )
        t2 = time.time()
        coef_hat = B[0]
        rs = np.array([
            p, l,
            "CD",
            (np.linalg.norm(coef_hat - coef)).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        # print("finished CD")

        # vanilla MIP
        t1 = time.time()
        obj, B, g_sol = vanilla_msetype(
            X_list, y_list, n_list=None, gamma1=5, rho=1.0, lam=0.01,
            s_max=None, beta_bound=100.0, S_start=inv_set, B_start=None,
            mip_focus=2, mip_cuts=2, forbidden=None, time_limit=180,
            mip_gap=None, verbose=False
        )
        t2 = time.time()
        coef_hat = B[0]
        inv_set = np.where(g_sol!=0)[0]
        rs = np.array([
            p, l,
            "MIP(warmStart)",
            (np.linalg.norm(coef_hat - coef)).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])

        # vanilla MIP 2
        t1 = time.time()
        obj, B, g_sol = vanilla_msetype(
            X_list, y_list, n_list=None, gamma1=5, rho=1.0, lam=0.01,
            s_max=None, beta_bound=100.0, S_start=(), B_start=None,
            mip_focus=2, mip_cuts=2, forbidden=None, time_limit=180,
            mip_gap=None, verbose=False
        )
        t2 = time.time()
        coef_hat = B[0]
        inv_set = np.where(g_sol!=0)[0]
        rs = np.array([
            p, l,
            "MIP",
            (np.linalg.norm(coef_hat - coef)).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
    
        # print(RS[(RS.shape[0]-2):RS.shape[0],:])
        # print("finished n=", n, "e=", E, "l=", l, '\n')
        pbar.update(1)

        print(RS[(RS.shape[0]-3):RS.shape[0],:])
        print('\n')

pbar.close()

# Preserve manuscript-independent result file locations.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
