import numpy as np
from invariant_prediction import cd_msetype,precompute_stat
from invariant_prediction.task2 import precompute_stat_test,cd_testtype
from invariant_prediction.bootstrap import train_test_split_envs,mip_test_multiple
from invariant_prediction.nonlinear import fourier_expand,precompute_stat as precompute_nonlinear,cd_msetype_grouped
from invariant_prediction.data import generate_synthetic_data_soft_intervention

def synthetic():
    np.random.seed(10)
    return generate_synthetic_data_soft_intervention(
        n=50,p=8,E=3,p_act=.2,lb=.1,ub=.2,lb_star=.5,ub_star=1,
        sigma_x_min=.5,sigma_x_max=1,sigma_y=.5,p_star_min=1,p_star_max=2,p_resample=.1
    )

def test_task1_cd():
    S,beta,X,y,n=synthetic()
    support,B,obj=cd_msetype(precompute_stat(X,y),lam=.01,gamma1=5,rho=1)
    assert B.shape==(len(X)+1,X[0].shape[1])
    assert all(0<=j<X[0].shape[1] for j in support)
    assert np.isfinite(obj)

def test_task2_cd():
    _,_,X,y,_=synthetic()
    support,B,obj=cd_testtype(precompute_stat_test(X,y),k0=0)
    assert 0 not in support
    assert np.isfinite(obj)

def test_nonlinear_group_cd():
    _,_,X,y,_=synthetic()
    Z,groups=fourier_expand(X,K=2)
    support,B,obj=cd_msetype_grouped(precompute_nonlinear(Z,y),groups,lam=.01,gamma1=.5,rho=1)
    assert set(support).issubset(set(range(X[0].shape[1])))
    assert np.isfinite(obj)

def test_split_and_bootstrap():
    _,_,X,y,_=synthetic()
    A,ya,na,B,yb,nb=train_test_split_envs(X,y,test_size=.8,seed=3)
    assert all(a+b==50 for a,b in zip(na,nb))
    selected=mip_test_multiple(A,ya,na,B,yb,nb,superset=[0,1],M_bar=1,B=5,alpha=.05,random_state=1,verbose=0)
    assert set(selected).issubset(set(range(X[0].shape[1])))


def test_baselines_sibling_import():
    from baselines.eills import fit_eills
    from baselines.bip import pi_analytical
    from invariant_prediction.experiment_api import fit_eills as experiment_eills
    assert callable(fit_eills) and callable(pi_analytical)
    assert experiment_eills is fit_eills


def test_wvm_local_screening():
    from baselines.wvm import DirectWassersteinVariance,WV_screening
    rng=np.random.default_rng(12)
    X=rng.normal(size=(40,3))
    y=X[:,0]+rng.normal(size=40)
    wv=DirectWassersteinVariance(X,y,[20,20],np.array([.5,.5]))
    np.random.seed(12)
    score,causes,obj,pvals,threshold=WV_screening(wv,B=3)
    assert score.shape==causes.shape==pvals.shape==(3,)
    assert np.isfinite(obj) and np.isfinite(threshold)


def test_loeo_uses_parameters():
    from invariant_prediction.utils import loeo_mse
    _,_,X,y,_=synthetic()
    lam,gamma,rho=.02,2.,.4
    expected=[]
    for e in range(len(X)):
        Xt=[x for i,x in enumerate(X) if i!=e]
        yt=[z for i,z in enumerate(y) if i!=e]
        _,B,_=cd_msetype(precompute_stat(Xt,yt),lam=lam,gamma1=gamma,rho=rho,S_init=())
        expected.append(np.mean((y[e]-X[e]@B[0])**2))
    assert np.isclose(loeo_mse(X,y,gamma1=gamma,rho=rho,lam=lam),np.mean(expected))

def test_experiment_parameter_grids():
    from pathlib import Path
    base=Path(__file__).resolve().parents[1]/'experiments'
    f2=(base/'fig02_solver_comparison'/'run.py').read_text()
    f3=(base/'fig03_identification'/'run.py').read_text()
    f7=(base/'fig07_baselines'/'run.py').read_text()
    f9=(base/'fig09_testing'/'run.py').read_text()
    assert 'p_grid = [10, 20, 30, 40, 50]' in f2 and 'n_trials = 100' in f2
    assert 'n_trials=50' in f3 and 'default=100' in f3
    assert 'range(200 if args.trials is None' in f7 and 'lam_best,gamma_best=min(grid,key=cv_score)' in f7
    assert 'range(3,18)' in f9 and 'range(200 if args.trials is None' in f9
