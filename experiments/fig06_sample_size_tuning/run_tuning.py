"""Figure 6 tuning experiment (Appendix G.3)."""
import argparse
from pathlib import Path
from invariant_prediction.experiment_api import *
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--trials',type=int,default=None)
parser.add_argument('--seed',type=int,default=None)
parser.add_argument('--output',default=None)
args=parser.parse_args()
if args.seed is not None: np.random.seed(args.seed)

from tqdm import tqdm
RS = np.zeros((0, 10))

gamma_grid = [1, 5, 10, 15, 20]
lam_grid = [0.0001, 0.001, 0.01, 0.1, 1]
N_trial = args.trials or 50

pbar = tqdm(total = len(gamma_grid) * len(lam_grid) * N_trial)

for gamma in gamma_grid:
    for lam in lam_grid:
        for l in range(N_trial):
            pbar.set_postfix(gamma=gamma, lam=lam, l=l)
            true_invariant, coef, X_list, y_list, n_list = generate_synthetic_data_soft_intervention(
                n=150,
                p=20,
                E=10,
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

            # vanilla MIP
            loeo_val = loeo_mse(X_list, y_list, gamma1=gamma, rho=1, lam=lam, s_max=None).item()

            # obj, B, g_sol = vanilla_msetype(
            #     X_list, y_list, n_list=None, gamma1=gamma, rho=1.0, lam=lam,
            #     s_max=None, beta_bound=100.0, S_start=(), B_start=None,
            #     mip_focus=2, mip_cuts=2, forbidden=None, time_limit=None,
            #     mip_gap=None, verbose=False
            # )
            t1 = time.time()
            stat = precompute_stat(X_list, y_list, n_list=None)
            inv_set, B, obj = cd_msetype(
                stat, lam=lam, gamma1=gamma, rho=1, forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
            )
            t2 = time.time()
            coef_hat = B[0]
            rs = np.array([
                gamma, lam, l,
                "MIP",
                (np.linalg.norm(coef_hat - coef)).item(),
                loeo_val,
                set(inv_set).issuperset(set(true_invariant)),
                set(inv_set) == set(true_invariant),
                len(inv_set),
                t2 - t1
            ])
            RS = np.vstack([RS, rs])

            pbar.update(1)
        
pbar.close()

output=Path(args.output) if args.output else HERE/'results'/'RS_reg_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
