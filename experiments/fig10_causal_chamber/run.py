"""Figure 10: light tunnel chamber comparisons."""
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

from pathlib import Path

# Settings
DATA_DIR = HERE/"data"/"lt_interventions_standard_v1"
N_TRIALS, N_PER_ENV, BASE_SEED = (args.trials or 100), 1000, 2026
E_VALUES = range(3, 11)

# Variables
FEATURES = [
    "red", "green", "blue", "pol_1", "pol_2",
    "vis_1", "vis_2", "vis_3", "ir_1", "ir_2", "current"
]
TARGET = "ir_3"
TRUE_SUPPORT = ["red", "green", "blue", "pol_1", "pol_2"]

# Environments
REFERENCE_ENV = "uniform_reference"

INTERVENTION_POOL = [
    "uniform_red_strong",
    "uniform_green_strong",
    "uniform_blue_strong",
    "uniform_t_vis_1_strong",
    "uniform_t_vis_2_strong",
    "uniform_t_vis_3_strong",
    "uniform_t_ir_1_strong",
    "uniform_t_ir_2_strong",
    "uniform_diode_vis_1_mid",
    "uniform_diode_vis_2_mid",
    "uniform_diode_vis_3_mid",
    "uniform_diode_ir_1_strong",
    "uniform_diode_ir_2_strong",
    "uniform_l_11_mid",
    "uniform_l_12_mid",
    "uniform_l_21_mid",
    "uniform_l_22_mid",
    "uniform_osr_c_strong",
    "uniform_v_c_strong",
]

# Read the first 500 observations from each environment
def read_environment(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv").iloc[:N_PER_ENV].reset_index(drop=True)
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {missing}")
    return df

data = {REFERENCE_ENV: read_environment(REFERENCE_ENV)}
data.update({env: read_environment(env) for env in INTERVENTION_POOL})

# Construct nested environment sets
def construct_trials(n_trials=N_TRIALS, e_values=E_VALUES, base_seed=BASE_SEED):
    X_trials, y_trials, env_trials = {}, {}, {}
    # Standardization based only on the reference environment
    ref = data[REFERENCE_ENV]
    x_mean = ref[FEATURES].mean()
    x_std = ref[FEATURES].std(ddof=0).replace(0, 1.0)
    y_mean = ref[TARGET].mean()
    y_std = ref[TARGET].std(ddof=0)

    for trial in range(n_trials):
        rng = np.random.default_rng(base_seed + trial)
        order = list(rng.permutation(INTERVENTION_POOL))
        X_trials[trial], y_trials[trial], env_trials[trial] = {}, {}, {}
        sampled_data = {
            env: data[env].iloc[rng.choice(len(data[env]), N_PER_ENV, replace=True)]
            for env in [REFERENCE_ENV] + INTERVENTION_POOL
        }
        for E in e_values:
            envs = [REFERENCE_ENV] + order[:E - 1]
            X_list = [
                ((data[env][FEATURES] - x_mean) / x_std).to_numpy(dtype=float)
                for env in envs
            ]
            y_list = [
                ((data[env][TARGET] - y_mean) / y_std).to_numpy(dtype=float)
                for env in envs
            ]
            X_trials[trial][E] = X_list
            y_trials[trial][E] = y_list
            env_trials[trial][E] = envs

    return X_trials, y_trials, env_trials

X_trials, y_trials, env_trials = construct_trials(n_trials=N_TRIALS)

TRUE_SUPPORT_IDX = np.array([0, 1, 2, 3, 4])

def selection_metrics(selected, true_support=TRUE_SUPPORT_IDX):
    selected, true_support = set(selected), set(true_support)
    tp = len(selected & true_support)
    fp = len(selected - true_support)
    fn = len(true_support - selected)
    # precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    # recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    # f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    # fdr = fp / max(1, len(selected))
    return {"tp": tp, "fp": fp, "fn": fn, "size": len(selected)}

RS = np.zeros((0, 8))

for l in range(N_TRIALS):
    for e in E_VALUES:

        X_list = X_trials[l][e]
        y_list = y_trials[l][e]
        n_list = [x.shape[0] for x in X_list]

        # MIP-cd
        t1 = time.time()
        stat = precompute_stat(X_list, y_list, n_list=None)
        mip_inv_set, B, obj = cd_msetype(
            stat, lam=0.01, gamma1=5, rho=1, forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
        )
        t2 = time.time()
        metrics = selection_metrics(mip_inv_set)
        rs = np.array([e, l, 'MIP', metrics['tp'], metrics['fp'], metrics['fn'], metrics['size'], t2-t1 ])
        RS = np.vstack([RS, rs])

        # EILLS
        t1 = time.time()
        what, g_eills = fit_eills(X_list, y_list, include_intercept=False, gamma=20)
        t2 = time.time()
        if len(g_eills) == 0:
            inv_set = []
        else:
            inv_set = g_eills
        eills_inv_set = inv_set
        metrics = selection_metrics(eills_inv_set)
        rs = np.array([e, l, 'EILLS', metrics['tp'], metrics['fp'], metrics['fn'], metrics['size'], t2-t1 ])
        RS = np.vstack([RS, rs])

        # ICP
        data = []
        for i in range(len(X_list)):
            data.append( np.column_stack([X_list[i], y_list[i]]) )
        target = X_list[0].shape[1]
        t1 = time.time()
        result = icp.fit(data, target, alpha=0.05, precompute=True, verbose=False, color=False)
        t2 = time.time()
        if len(result.accepted_sets) ==0:
            intersection=[]
        else:
            intersection = reduce(lambda a, b: a & b, result.accepted_sets)
        icp_inv_set = intersection
        metrics = selection_metrics(icp_inv_set)
        rs = np.array([e, l, 'ICP', metrics['tp'], metrics['fp'], metrics['fn'], metrics['size'], t2-t1 ])
        RS = np.vstack([RS, rs])

        # Causal Dantzig
        X = np.vstack(X_list)      # shape (n_total, p)
        Y = np.concatenate(y_list) # shape (n_total,)
        ExpInd = []
        for i, Xi in enumerate(X_list):
            n_i = Xi.shape[0]
            ExpInd.extend([i]*n_i)
        ExpInd = np.array(ExpInd)  # length n_total
        t1 = time.time()
        dantzig_inv_set = np.where( fit_hiddenICP(X, Y, ExpInd, alpha=0.05, intercept=False)['maximinCoefficients'] != 0 )[0]
        t2 = time.time()
        metrics = selection_metrics(dantzig_inv_set)
        rs = np.array([e, l, 'Causal Dantzig', metrics['tp'], metrics['fp'], metrics['fn'], metrics['size'], t2-t1 ])
        RS = np.vstack([RS, rs])

        
        # BIP
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
        bip_inv_set = np.where(map_z !=0 )[0]
        metrics = selection_metrics(bip_inv_set)
        rs = np.array([e, l, 'BIP', metrics['tp'], metrics['fp'], metrics['fn'], metrics['size'], t2-t1 ])
        RS = np.vstack([RS, rs])

        # MIP-test
        t1 = time.time()
        X_mip_list, y_mip_list, n_mip_list, X_diag_list, y_diag_list, n_diag_list = train_test_split_envs(
            X_list, y_list, test_size=.8
        )
        stat = precompute_stat(X_mip_list, y_mip_list, n_list=None)
        mip_inv_set, B, obj = cd_msetype(
            stat, lam=0.01, gamma1=5, rho=1, forbid=None, S_init=(), superset_init=(), tol=1e-10, max_iter=100
        )
        S_test = mip_inv_set
        X_list_screen = []
        for e_ in range(len(X_diag_list)):
            X_e_screen = X_diag_list[e_][:, S_test]  # select features
            X_list_screen.append(X_e_screen)
        X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list = train_test_split_envs(
            X_list_screen, y_diag_list, test_size=.8
        )
        inv_set = mip_test_multiple(
            X_train_list, y_train_list, n_train_list, X_test_list, y_test_list, n_test_list, 
            superset=[j for j in range(X_train_list[0].shape[1])], gamma=(1,1,1,1), alpha=.05,
            M_bar=5, B=500, solver="cd", verbose=0,
        )
        t2 = time.time()
        MIPtest_inv_set = np.array(mip_inv_set)[inv_set]
        metrics = selection_metrics(MIPtest_inv_set)
        rs = np.array([e, l, 'MIPtest', metrics['tp'], metrics['fp'], metrics['fn'], metrics['size'], t2-t1 ])
        RS = np.vstack([RS, rs])

        # FAIR
        t1 = time.time()
        fair_inv_set, probs, gate_history = fit_fair_nn(X_list, y_list, gamma=36, niters=10000, threshold=.9)
        t2 = time.time()
        metrics = selection_metrics(fair_inv_set)
        rs = np.array([e, l, 'FAIR', metrics['tp'], metrics['fp'], metrics['fn'], metrics['size'], t2-t1 ])
        RS = np.vstack([RS, rs])

        print('finished', 'l=', l, 'e=', e)
        print(RS[(RS.shape[0]-6):RS.shape[0],:])

# Keep figure data and generated outputs in separate files.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(RS).to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
