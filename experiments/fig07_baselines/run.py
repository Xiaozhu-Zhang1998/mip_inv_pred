"""Figure 7: comparison with competing methods (Appendix G.4)."""
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
    torch.manual_seed(args.seed)

# strong interventions

RS = np.zeros((0, 8))

for p in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]:
    for l in range(200 if args.trials is None else args.trials):
        n=500
        E_train=10
        if int(0.4 * p) < 16:   
            p_star = int(0.4 * p)
        else:
            p_star = 16 + int(0.1 * p)
        true_invariant, coef, X_list, y_list, n_list = generate_synthetic_data_all_intervention(
            n=n,
            p=p,
            E=E_train,
            p_act=np.random.choice([0.1,0.15]),
            lb=0.2, ub=0.5,
            lb_star=0.5, ub_star=0.5,
            sigma_x_min=0.5,
            sigma_x_max=1.0,
            sigma_y=0.5,
            p_star_min=p_star,
            p_star_max=p_star
        )
        super_structure_10 = ridge_screening(X_list, y_list, 10, lam=1.0, intercept=False)[0]
        super_structure_15 = ridge_screening(X_list, y_list, 15, lam=1.0, intercept=False)[0]

        # MIP-CD: select parameters by LOEO, then time the final fit
        folds=[]
        for e in range(len(X_list)):
            stat_e=precompute_stat([X_list[k] for k in range(len(X_list)) if k!=e],
                                   [y_list[k] for k in range(len(y_list)) if k!=e])
            folds.append((stat_e,X_list[e],y_list[e]))
        def cv_score(par):
            lam,gamma=par
            errors=[]
            for stat_e,X_val,y_val in folds:
                _,B_e,_=cd_msetype(stat_e,lam=lam,gamma1=gamma,rho=1,S_init=())
                errors.append(np.mean((y_val-X_val@B_e[0])**2))
            return np.mean(errors)
        grid=((lam,gamma) for lam in (1e-4,1e-3,1e-2,1e-1,1) for gamma in (1,5,10,15,20))
        lam_best,gamma_best=min(grid,key=cv_score)
        t1=time.time()
        stat=precompute_stat(X_list,y_list)
        inv_set,B,obj=cd_msetype(stat,lam=lam_best,gamma1=gamma_best,rho=1,S_init=())
        t2 = time.time()
        coef_hat = B[0]
        rs = np.array([
            p,l,
            "MIP-CD",
            (np.linalg.norm(coef_hat - coef)).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished MIP")
        

        # EILLS
        X_list_screen = []
        for e in range(len(X_list)):
            X_list_screen.append( X_list[e][:,super_structure_15] )
        t1 = time.time()
        coef_eills, g_eills = fit_eills(X_list_screen, y_list, include_intercept=False, gamma=20)
        t2 = time.time()
        if len(g_eills) == 0:
            inv_set = []
        else:
            inv_set = super_structure_15[g_eills] 
        coef_hat = np.zeros(p)
        coef_hat[super_structure_15] = coef_eills
        rs = np.array([
            p,l,
            "EILLS-s",
            (np.linalg.norm(coef_hat - coef)).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished EILLS")

        # ICP
        X_list_screen = []
        for e in range(len(X_list)):
            X_list_screen.append( X_list[e][:,super_structure_10] )
        data = []
        for i in range(len(X_list_screen)):
            data.append( np.column_stack([X_list_screen[i], y_list[i]]) )
        target = X_list_screen[0].shape[1]
        t1 = time.time()
        result = icp.fit(data, target, alpha=0.05, precompute=True, verbose=False, color=False)
        t2 = time.time()
        if len(result.accepted_sets) == 0:
            intersection = []
        else:
            intersection = reduce(lambda a, b: a & b, result.accepted_sets)
        inv_set = super_structure_10[list(intersection)]
        rs = np.array([
            p,l,
            "ICP-s",
            np.nan,
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished ICP")

        # Hidden-ICP
        X = np.vstack(X_list)      # shape (n_total, p)
        Y = np.concatenate(y_list) # shape (n_total,)
        ExpInd = []
        start = 0
        for i, Xi in enumerate(X_list):
            n_i = Xi.shape[0]
            ExpInd.extend([i]*n_i)
        ExpInd = np.array(ExpInd)  # length n_total
        t1 = time.time()
        result = fit_hiddenICP(X, Y, ExpInd, alpha=0.1, intercept=False)['maximinCoefficients']
        t2 = time.time()
        inv_set = np.where( result != 0 )[0]
        coef_hat = result
        rs = np.array([
            p,l,
            "Causal Dantzig",
            (np.linalg.norm(coef_hat - coef)).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished Dantzig")

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
            p_max=10,     # enumerate subsets over the ten screened predictors
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
            p,l,
            "BIP-s",
            (np.linalg.norm(coef_hat - coef)).item(),
            set(inv_set).issuperset(set(true_invariant)),
            set(inv_set) == set(true_invariant),
            len(inv_set),
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished BIP")

        
        # NegDRO
        t1 = time.time()
        b_final, _ = negDRO(X_list, y_list, gamma = 20)
        t2 = time.time()
        coef_hat = b_final
        rs = np.array([
            p,l,
            "NegDRO",
            (np.linalg.norm(coef_hat - coef)).item(),
            np.nan,
            np.nan,
            np.nan,
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished NegDRO")

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
            p,l,
            "CoCo",
            (np.linalg.norm(coef_hat - coef)).item(),
            np.nan,
            np.nan,
            np.nan,
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished CoCo")

        # IRM
        t1 = time.time()
        coef_hat = irm_linear(
            X_list, y_list,
            penalty_weight=10,
            penalty_anneal_iters=100,
            l2_weight=1e-3,
            lr=1e-3,
            steps=500
        )
        t2 = time.time()
        rs = np.array([
            p,l,
            "IRM",
            (np.linalg.norm(coef_hat - coef)).item(),
            np.nan,
            np.nan,
            np.nan,
            t2 - t1
        ])
        RS = np.vstack([RS, rs])
        print("finished IRM")
    
        print(RS[(RS.shape[0]-8):RS.shape[0],:])
        print("finished p=", p, "l=", l, '\n')

# Keep figure data and generated outputs in separate files.
output=Path(args.output) if args.output else HERE/'results'/'RS_generated.csv'
output.parent.mkdir(parents=True,exist_ok=True)
df=pd.DataFrame(RS)
pvals=pd.to_numeric(df.iloc[:,0]).astype(int)
df['true_s']=pvals.apply(lambda p: int(.4*p) if int(.4*p)<16 else 16+int(.1*p))
df.to_csv(output,index=False,header=False)
print(f"Saved {len(RS)} rows to {output}")
