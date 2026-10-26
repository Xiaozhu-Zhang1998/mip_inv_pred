"""Generators for the synthetic experiments and manuscript examples."""
import numpy as np

# From original notebook, cell 77
def generate_synthetic_data_all_intervention(
    n, p, E, p_act, lb, ub, lb_star, ub_star,
    sigma_x_min, sigma_x_max, sigma_y,
    p_star_min, p_star_max, pos_y=None
):
    total = p + 1
    pos_y = total // 2 if pos_y is None else pos_y

    # Common topological order
    perm = np.random.permutation(total)
    i_y = np.where(perm == p)[0][0]
    perm[i_y], perm[pos_y] = perm[pos_y], perm[i_y]

    # Invariant parent set and coefficients of Y
    candidates = perm[:pos_y]
    s = np.random.randint(p_star_min, min(p_star_max, len(candidates)) + 1)
    S_star = np.sort(np.random.choice(candidates, s, replace=False))
    beta_star = np.zeros(p)
    beta_star[S_star] = (
        np.random.choice([-1, 1], s)
        * np.random.uniform(lb_star, ub_star, s)
    )
    X_list, y_list = [], []

    for _ in range(E):
        alpha = np.zeros((total, total))
        # Environment-specific X mechanisms; invariant Y mechanism
        for i in range(total):
            child = perm[i]
            for parent in perm[:i]:
                if child == p:
                    if parent in S_star: 
                        alpha[child, parent] = beta_star[parent]
                elif np.random.rand() < p_act:
                    alpha[child, parent] = (
                        np.random.choice([-1, 1])
                        * np.random.uniform(lb, ub)
                    )
        sigma_x = np.random.uniform(sigma_x_min, sigma_x_max, p)
        data = np.zeros((n, total))
        for t in range(n):
            values = np.zeros(total)
            for node in perm:
                sd = sigma_y if node == p else sigma_x[node]
                values[node] = alpha[node] @ values + np.random.normal(0, sd)
            data[t] = values
        X_list.append(data[:, :p])
        y_list.append(data[:, p])

    return S_star, beta_star, X_list, y_list, [n] * E

# From original notebook, cell 79
def generate_synthetic_data_soft_intervention(
    n, p, E, p_act, lb, ub, lb_star, ub_star,
    sigma_x_min, sigma_x_max, sigma_y,
    p_star_min, p_star_max, p_resample=1.0,
    pos_y=None, intervene_on_Y=False, random_state=None
):
    rng = np.random.default_rng(random_state)
    coef = lambda a, b: rng.choice([-1., 1.]) * rng.uniform(a, b)

    total = p + 1
    pos_y = total // 2 if pos_y is None else pos_y
    if not 0 <= pos_y < total: raise ValueError("pos_y must be in {0,...,p}.")

    # Common topological order
    perm = rng.permutation(total)
    i_y = np.where(perm == p)[0][0]
    perm[i_y], perm[pos_y] = perm[pos_y], perm[i_y]

    # Invariant parents and coefficients of Y
    candidates = perm[:pos_y]
    if len(candidates) == 0: raise ValueError("No candidates available before Y.")

    s_min, s_max = min(p_star_min, len(candidates)), min(p_star_max, len(candidates))
    if s_min < 1: raise ValueError("Need at least one parent candidate for Y.")

    s = rng.integers(s_min, s_max + 1)
    S_star = np.sort(rng.choice(candidates, s, replace=False))

    beta_star = np.zeros(p)
    beta_star[S_star] = [coef(lb_star, ub_star) for _ in S_star]

    # Fixed DAG support
    support = np.zeros((total, total), int)
    support[p, S_star] = 1

    for i, child in enumerate(perm):
        if child == p: continue
        for parent in perm[:i]:
            if rng.random() < p_act: support[child, parent] = 1

    # Baseline coefficients
    alpha_base = np.zeros((total, total))
    alpha_base[p, S_star] = beta_star[S_star]

    for child in range(p):
        for parent in np.where(support[child])[0]:
            alpha_base[child, parent] = coef(lb, ub)

    # Same noise distributions in every environment
    sigma_x = rng.uniform(sigma_x_min, sigma_x_max, p)

    X_list, y_list = [], []

    for e in range(E):
        alpha = alpha_base.copy()

        if e > 0:
            for child in range(p):
                for parent in np.where(support[child])[0]:
                    if rng.random() < p_resample:
                        alpha[child, parent] = coef(lb, ub)

            if intervene_on_Y:
                for parent in S_star:
                    if rng.random() < p_resample:
                        alpha[p, parent] = coef(lb_star, ub_star)

        data = np.zeros((n, total))

        for t in range(n):
            v = np.zeros(total)
            for node in perm:
                sd = sigma_y if node == p else sigma_x[node]
                v[node] = alpha[node] @ v + rng.normal(0, sd)
            data[t] = v

        X_list.append(data[:, :p])
        y_list.append(data[:, p])

    return S_star, beta_star, X_list, y_list, [n] * E

# From original notebook, cell 81
def generate_X1_to_Y_to_X2_example(
    n_list=(1000, 1000),
    k_independent=0,
    gamma=1.0,
    tau2=1.0,
    sigma_eps2=1.0,
    a_list=(1.0, 0.5),
    v_list=(2.0, 1.5),
    z_var=1.0,
    seed=None,
    return_params=False,
):
    """
    Generate multi-environment data from the SEM

        X1^e = N1^e
        Y^e  = gamma * X1^e + eps^e
        X2^e = a_e * Y^e + eta^e

    plus k_independent independent noise features

        Z_1^e, ..., Z_k^e,

    where the Z variables are independent of X1, Y, X2 and all structural noises.

    The default two-environment parameters are chosen so that for S = {X2},

        beta^{1,S} = beta^{2,S},

    but

        sigma^{1,S} != sigma^{2,S}.

    Parameters
    ----------
    n_list : tuple or list of int
        Sample sizes for each environment.

    k_independent : int
        Number of additional independent noise features.

    gamma : float
        Structural coefficient from X1 to Y.

    tau2 : float
        Variance of X1.

    sigma_eps2 : float
        Variance of eps in the Y equation.

    a_list : tuple or list of float
        Coefficients from Y to X2 in each environment.

    v_list : tuple or list of float
        Noise variances of eta in each environment.

    z_var : float
        Variance of each independent noise feature.

    seed : int, optional
        Random seed.

    return_params : bool
        If True, also return the parameter dictionary.

    Returns
    -------
    X_list : list of np.ndarray
        X_list[e] has shape (n_e, 2 + k_independent).
        Column 0 is X1.
        Column 1 is X2.
        Columns 2 onward are independent noise features.

    y_list : list of np.ndarray
        y_list[e] has shape (n_e,).

    n_list : list of int
        Sample sizes.

    params : dict, optional
        Returned only if return_params=True.
    """

    rng = np.random.default_rng(seed)

    n_list = list(n_list)
    a_list = list(a_list)
    v_list = list(v_list)

    E = len(n_list)

    if len(a_list) != E:
        raise ValueError("a_list must have the same length as n_list.")
    if len(v_list) != E:
        raise ValueError("v_list must have the same length as n_list.")
    if k_independent < 0:
        raise ValueError("k_independent must be nonnegative.")
    if z_var < 0:
        raise ValueError("z_var must be nonnegative.")

    X_list = []
    y_list = []

    for e in range(E):
        n = int(n_list[e])
        a_e = a_list[e] + rng.normal(loc=0.0, scale=0.05, size=1)
        v_e = v_list[e] + rng.normal(loc=0.0, scale=0.05, size=1)

        X1 = rng.normal(loc=0.0, scale=np.sqrt(tau2), size=n)
        eps = rng.normal(loc=0.0, scale=np.sqrt(sigma_eps2), size=n)
        eta = rng.normal(loc=0.0, scale=np.sqrt(v_e), size=n)

        y = gamma * X1 + eps
        X2 = a_e * y + eta

        if k_independent > 0:
            Z = rng.normal(
                loc=0.0,
                scale=np.sqrt(z_var),
                size=(n, k_independent),
            )
            X = np.column_stack([X1, X2, Z])
        else:
            X = np.column_stack([X1, X2])

        X_list.append(X)
        y_list.append(y)

    if not return_params:
        return X_list, y_list, n_list

    c = gamma**2 * tau2 + sigma_eps2

    params = {
        "gamma": gamma,
        "tau2": tau2,
        "sigma_eps2": sigma_eps2,
        "a_list": a_list,
        "v_list": v_list,
        "k_independent": k_independent,
        "z_var": z_var,
        "feature_names": ["X1", "X2"] + [f"Z{j + 1}" for j in range(k_independent)],
        "true_parent_set": [0],
        "bad_set_equal_beta_unequal_sigma": [1],
        "independent_feature_set": list(range(2, 2 + k_independent)),
        "population_beta_on_X2": [
            (a_list[e] * c) / (a_list[e] ** 2 * c + v_list[e])
            for e in range(E)
        ],
        "population_residual_var_on_X2": [
            c - (a_list[e] * c) ** 2 / (a_list[e] ** 2 * c + v_list[e])
            for e in range(E)
        ],
    }

    return X_list, y_list, n_list, params

# From original notebook, cell 83
def generate_multiple_invariant(
    n_train, n_test, k=5, noise_std=.5,
    lbd=.5, ubd=2., perturb_strength=3, seed=None
):
    """
    SEM:
        X2 -> X3
        X1 -> Y <- X3
        Y  -> X4

    Environments:
        0: observational
        1: intervene on X1
        2: intervene on X2

    True parent set of Y:
        S* = {X1, X3}
    """
    rng = np.random.default_rng(seed)

    a3 = rng.uniform(lbd, ubd)   # X2 -> X3
    b1 = rng.uniform(lbd, ubd)   # X1 -> Y
    b3 = rng.uniform(lbd, ubd)   # X3 -> Y
    c4 = rng.uniform(lbd, ubd)   # Y  -> X4

    coef = np.zeros(4+k)
    coef[0], coef[2] = b1, b3

    def generate_env_data(n, env):
        X1 = rng.normal(0, 1, n)
        X2 = rng.normal(0, 1, n)

        if env == 1:
            X1 = rng.choice([-perturb_strength, perturb_strength], n)
        elif env == 2:
            X2 = rng.choice([-perturb_strength, perturb_strength], n)

        e3 = rng.normal(0, noise_std, n)
        ey = rng.normal(0, noise_std, n)
        e4 = rng.normal(0, noise_std, n)

        X3 = a3*X2 + e3
        Y  = b1*X1 + b3*X3 + ey
        X4 = c4*Y + e4

        X = np.column_stack([X1, X2, X3, X4])
        if k > 0: X = np.column_stack([X, rng.normal(0, 1, (n, k))])

        return X, Y

    X_train_list, y_train_list, n_train_list = [], [], []
    X_test_list, y_test_list, n_test_list = [], [], []

    for env in range(3):
        Xtr, ytr = generate_env_data(n_train, env)
        Xte, yte = generate_env_data(n_test, env)

        X_train_list.append(Xtr); y_train_list.append(ytr); n_train_list.append(n_train)
        X_test_list.append(Xte); y_test_list.append(yte); n_test_list.append(n_test)

    return (
        X_train_list, y_train_list, n_train_list,
        X_test_list, y_test_list, n_test_list, coef
    )

# From original notebook, cell 85
def generate_hidden_sem_data(
    n, E, p_a=None, p_s=4, p_d=None, k=8, q=2,
    beta=None, Gamma_A=None, B_AS=None, Gamma_S=None,
    Gamma_D=None, phi_Y=None, theta_D=None,
    a_scales=None, s_scales=None, d_scales=None,
    sigma_y=.2, gamma_a_scale=.3, gamma_s_scale=.3,
    gamma_d_scale=.3, b_as_scale=.3, seed=None
):
    rng = np.random.default_rng(seed)

    p_a = rng.choice([2,3]) if p_a is None else int(p_a)
    p_d = rng.choice([2,3]) if p_d is None else int(p_d)
    n_list = [int(n)]*E if np.isscalar(n) else list(map(int, n))
    if len(n_list) != E: raise ValueError("n must have length E.")

    beta = rng.normal(0,1,p_s) if beta is None else np.asarray(beta,float)
    Gamma_A = rng.normal(0,gamma_a_scale,(q,p_a)) if Gamma_A is None else np.asarray(Gamma_A,float)
    B_AS = rng.normal(0,b_as_scale,(p_a,p_s)) if B_AS is None else np.asarray(B_AS,float)
    Gamma_S = rng.normal(0,gamma_s_scale,(q,p_s)) if Gamma_S is None else np.asarray(Gamma_S,float)
    Gamma_D = rng.normal(0,gamma_d_scale,(q,p_d)) if Gamma_D is None else np.asarray(Gamma_D,float)
    phi_Y = rng.normal(0,1,q) if phi_Y is None else np.asarray(phi_Y,float)
    theta_D = rng.normal(0,1,p_d) if theta_D is None else np.asarray(theta_D,float)

    if beta.shape != (p_s,): raise ValueError("beta has wrong shape.")
    if Gamma_A.shape != (q,p_a): raise ValueError("Gamma_A has wrong shape.")
    if B_AS.shape != (p_a,p_s): raise ValueError("B_AS has wrong shape.")
    if Gamma_S.shape != (q,p_s): raise ValueError("Gamma_S has wrong shape.")
    if Gamma_D.shape != (q,p_d): raise ValueError("Gamma_D has wrong shape.")
    if phi_Y.shape != (q,): raise ValueError("phi_Y has wrong shape.")
    if theta_D.shape != (p_d,): raise ValueError("theta_D has wrong shape.")

    a_scales = [1+.1*e for e in range(E)] if a_scales is None else list(a_scales)
    s_scales = [1+.1*e for e in range(E)] if s_scales is None else list(s_scales)
    d_scales = [1+.1*e for e in range(E)] if d_scales is None else list(d_scales)
    if not all(len(x) == E for x in [a_scales,s_scales,d_scales]):
        raise ValueError("Noise-scale lists must have length E.")

    X_list, y_list = [], []

    for e, ne in enumerate(n_list):
        H = rng.normal(0,1,(ne,q))

        X_A = H@Gamma_A + rng.normal(0,a_scales[e],(ne,p_a))
        X_S = X_A@B_AS + H@Gamma_S + rng.normal(0,s_scales[e],(ne,p_s))
        y = X_S@beta + H@phi_Y + rng.normal(0,sigma_y,ne)
        X_D = y[:,None]*theta_D + H@Gamma_D + rng.normal(0,d_scales[e],(ne,p_d))
        X_I = rng.normal(0,1,(ne,k))

        X_list.append(np.c_[X_A, X_S, X_D, X_I])
        y_list.append(y)

    A = np.arange(p_a)
    S = np.arange(p_a, p_a+p_s)
    D = np.arange(p_a+p_s, p_a+p_s+p_d)
    I = np.arange(p_a+p_s+p_d, p_a+p_s+p_d+k)

    true_coef = np.zeros(p_a+p_s+p_d+k)
    true_coef[S] = beta

    params = {
        "Gamma_A": Gamma_A,
        "B_AS": B_AS,
        "Gamma_S": Gamma_S,
        "Gamma_D": Gamma_D,
        "phi_Y": phi_Y,
        "theta_D": theta_D,
        "a_scales": np.asarray(a_scales),
        "s_scales": np.asarray(s_scales),
        "d_scales": np.asarray(d_scales)
    }

    return X_list, y_list, n_list, A, S, D, I, true_coef, params

# From original notebook, cell 87
def generate_synthetic_data_all_intervention_nonlinear(
    n, p, E, p_act, lb, ub, lb_star, ub_star,
    sigma_x_min, sigma_x_max, sigma_y,
    p_star_min, p_star_max, pos_y=None,
    random_state=None
):
    rng = np.random.default_rng(random_state)
    coef = lambda a, b: rng.choice([-1., 1.]) * rng.uniform(a, b)

    total = p + 1
    pos_y = total // 2 if pos_y is None else pos_y
    if not 0 <= pos_y < total: raise ValueError("pos_y must be in {0,...,p}.")

    # Common topological order
    perm = rng.permutation(total)
    i_y = np.where(perm == p)[0][0]
    perm[i_y], perm[pos_y] = perm[pos_y], perm[i_y]

    # Invariant parent set and nonlinear mechanism of Y
    candidates = perm[:pos_y]
    s_min = min(p_star_min, len(candidates))
    s_max = min(p_star_max, len(candidates))
    if s_min < 1: raise ValueError("Need at least one parent candidate for Y.")

    s = rng.integers(s_min, s_max + 1)
    S_star = np.sort(rng.choice(candidates, s, replace=False))

    beta_star = np.zeros(p)
    beta_star[S_star] = [coef(lb_star, ub_star) for _ in S_star]

    X_list, y_list = [], []

    for _ in range(E):
        alpha = np.zeros((total, total))

        # Environment-specific X mechanisms; invariant Y mechanism
        for i, child in enumerate(perm):
            for parent in perm[:i]:
                if child == p:
                    if parent in S_star:
                        alpha[child, parent] = beta_star[parent]
                elif rng.random() < p_act:
                    alpha[child, parent] = coef(lb, ub)

        # Environment-specific X noise scales
        sigma_x = rng.uniform(sigma_x_min, sigma_x_max, p)

        data = np.zeros((n, total))

        for t in range(n):
            v = np.zeros(total)

            for node in perm:
                pa = np.flatnonzero(alpha[node])
                mu = np.sum(alpha[node, pa] * np.tanh(2 * np.sin(v[pa]))) if len(pa) else 0.
                sd = sigma_y if node == p else sigma_x[node]
                v[node] = mu + rng.normal(0, sd)

            data[t] = v

        X_list.append(data[:, :p])
        y_list.append(data[:, p])

    return S_star, beta_star, X_list, y_list, [n] * E

