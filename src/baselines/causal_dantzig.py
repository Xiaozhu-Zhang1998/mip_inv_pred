"""Baseline implementation from the supplied research notebook. See docs/baselines.md."""
import numpy as np
from scipy import stats

# From original notebook, cell 59
# causal Danztig (aka hiddenICP) implementation in Python
# Code translated by ChatGPT from the Invariant Causal Prediction R package https://cran.r-project.org/web/packages/InvariantCausalPrediction/index.html
from scipy.stats import t, norm

def fit_hiddenICP(X, Y, ExpInd, alpha=0.05, mode="asymptotic", intercept=False):
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape(-1, 1)
    if not isinstance(X, np.ndarray):
        raise ValueError("'X' must be a numpy array")
    if not isinstance(Y, np.ndarray):
        raise ValueError("'Y' must be a numpy array")
    if X.shape[0] <= X.shape[1]:
        raise ValueError(f"hiddenICP not suitable for high-dimensional data (at the moment) -- need nrow(X)>ncol(X) but have nrow(X)={X.shape[0]} and ncol(X)={X.shape[1]}")
    
    if not isinstance(ExpInd, list):
        if len(ExpInd) != len(Y):
            raise ValueError("if `ExpInd' is a vector, it needs to have the same length as `Y'")
        unique_env = np.unique(ExpInd)
        if len(unique_env) == 1:
            raise ValueError(f"there is just one environment ('ExpInd'={unique_env[0]} for all observations) and the method needs at least two distinct environments")
        if min(np.bincount(ExpInd)) <= 2:
            print("\n out put of 'table(ExpInd)':\n", np.bincount(ExpInd))
            raise ValueError("one environment has just one or two observations (as supplied by 'ExpInd'); there need to be at least 3 (and ideally dozens) of observations in each environment")
        
        K = len(unique_env)
        ExpIndNEW = []
        for uc in unique_env:
            ExpIndNEW.append(np.where(ExpInd == uc)[0])
        ExpInd = ExpIndNEW
    else:
        flat_indices = [idx for sublist in ExpInd for idx in sublist]
        if min(flat_indices) < 0:
            raise ValueError(f"if `ExpInd' is a list with indices of observations, minimal entry has to be at least 0 but is {min(flat_indices)}")
        if max(flat_indices) >= len(Y):
            raise ValueError(f"if `ExpInd' is a list with indices of observations, maximal entry has to be at most equal to {len(Y)-1} of the observations but is {max(flat_indices)}")
    
    colX = [f"Variable_{i}" for i in range(X.shape[1])]

    if intercept:
        # NOTE that the intercept is the last column
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    K = len(ExpInd)
    n, p = X.shape

    KC = K if K > 2 else 1
    ConfInt = np.full((2, p), np.nan)
    pvalues = np.ones(p)
    
    betahatall = np.zeros(p)
    
    for kc in range(KC):
        ins = ExpInd[kc]
        out = np.setdiff1d(np.arange(n), ins)
        
        DS = (X[ins].T @ X[ins]) / len(ins) - (X[out].T @ X[out]) / len(out) #  (p, p)
        Drho = (X[ins].T @ Y[ins]) / len(ins) - (X[out].T @ Y[out]) / len(out) # (p,)
        try:
            DSI = np.linalg.inv(DS) # (p, p)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Matrix inversion failed: {e}")
        
        betahat = np.linalg.solve(DS, Drho) # (p,)
        betahatall += betahat
        
        
        Zin = np.zeros((len(ins), p))
        Zout = np.zeros((len(out), p))
        
        for i in range(len(ins)):
            tmp = DSI @ X[ins[i]] # (p,p) @ (p,) = (p,)
            Zin[i] = -tmp * np.sum(tmp * Drho) + Y[ins[i]] * tmp
        for i in range(len(out)):
            tmp = DSI @ X[out[i]]
            Zout[i] = -tmp * np.sum(tmp * Drho) + Y[out[i]] * tmp

        sigmavec = np.sqrt(np.diag(np.cov(Zin, rowvar=False) / len(ins) + np.cov(Zout, rowvar=False) / len(out)))
        
        pvalues = np.minimum(pvalues, 2 * K * (1 - t.cdf(np.abs(betahat) / np.maximum(1e-10, sigmavec), df=n - 1)))
        
        addvar = norm.ppf(max(0.5, 1 - alpha / (2 * K))) * sigmavec
        maximineffectsN = np.sign(betahat) * np.maximum(0, np.abs(betahat) - addvar)
        
        # Update lower bounds of confidence intervals
        new_lower_bounds = np.nan_to_num(betahat - addvar, nan=-np.inf)
        if kc == 0:
            ConfInt[0] = new_lower_bounds
        else:
            ConfInt[0] = np.minimum(ConfInt[0], new_lower_bounds)
        
        new_upper_bounds = np.nan_to_num(betahat + addvar, nan=np.inf)
        if kc == 0:
            ConfInt[1] = new_upper_bounds
        else:
            ConfInt[1] = np.maximum(ConfInt[1], new_upper_bounds)
        
        if kc == 0:
            maximineffects = maximineffectsN
        else:
            maximineffects = np.where(np.abs(maximineffectsN) > np.abs(maximineffects), maximineffectsN, maximineffects)
    
    betahat = betahatall / KC
    maximinCoefficients = maximineffects
    ConfInt = np.sort(ConfInt, axis=0) #[::-1]
    
    retobj = {
        "betahat": betahat,
        "maximinCoefficients": maximinCoefficients,
        "ConfInt": ConfInt,
        "pvalues": pvalues,
        "colnames": colX,
        "alpha": alpha
    }
    
    return retobj

