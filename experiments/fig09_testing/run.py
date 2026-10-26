"""Figure 9: testing versus number of environments (Appendix G.6)."""
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

p = 10
n = 500

RS = np.zeros((0, 7))

for E in range(3,18):
    for l in range(200 if args.trials is None else args.trials):
        true_invariant, coef, X_list, y_list, n_list = generate_synthetic_data_all_intervention(
            n=n,
            p=p,
            E=E,
            p_act=np.random.choice([0.1,0.15]),
            lb=0.2, ub=0.5,
            lb_star=0.5, ub_star=0.5,
            sigma_x_min=0.5,
            sigma_x_max=1.0,
            sigma_y=0.5,
            p_star_min=4,
            p_star_max=5
        )

        X = np.vstack(X_list)          # shape: (sum(n_list), p)
        y = np.concatenate(y_list)     # shape: (sum(n_list),)
        n_es = n_list                  # e.g. [500, 500, 500]
        n_total = sum(n_es)
        
        X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list = train_test_split_envs(
            X_list, y_list, test_size=.8
        )

        ## MIP-2
        t1 = time.time()
        inv_set = mip_test_multiple(
            X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list, 
            superset=[j for j in range(p)], gamma=(1,1,0,0), alpha=.05,
            M_bar=3, B=500, solver="cd", verbose=0,
        )
        t2 = time.time()
        rs = np.array([
            E, l, "MIP-test-2", 
            set(inv_set).issubset(set(true_invariant)),
            len(set(inv_set) & set(true_invariant)),
            set(inv_set) == set(true_invariant),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished MIP-2")

        
        ## MIP-4
        t1 = time.time()
        inv_set = mip_test_multiple(
            X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list, 
            superset=[j for j in range(p)], gamma=(1,1,1,1), alpha=.05,
            M_bar=3, B=500, solver="cd", verbose=0,
        )
        t2 = time.time()
        inv_set
        rs = np.array([
            E, l, "MIP-test-4", 
            set(inv_set).issubset(set(true_invariant)),
            len(set(inv_set) & set(true_invariant)),
            set(inv_set) == set(true_invariant),
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
            E, l, "ICP", 
            set(inv_set).issubset(set(true_invariant)),
            len(set(inv_set) & set(true_invariant)),
            set(inv_set) == set(true_invariant),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished ICP")

        ### Wasserstein
        t1 = time.time()
        wv = DirectWassersteinVariance(X, y, n_es, np.array(n_es) / n_total)
        _, causes, _, _, _ = WV_screening(wv, alpha=0.05, B=300)
        t2 = time.time()
        inv_set = np.where(causes == True)[0]
        rs = np.array([
            E, l, "WVM", 
            set(inv_set).issubset(set(true_invariant)),
            len(set(inv_set) & set(true_invariant)),
            set(inv_set) == set(true_invariant),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished Wasserstein")
        
        print("finished E=", E, "l=", l)

# Keep figure data and generated outputs in separate files.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
