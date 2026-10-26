# External baseline methods

The repository contains **no source copies** of the downloaded `bip`, `FAIR`, or `WVM` repositories. Their adapters in `src/baselines/` import code from separate installations when needed. To reproduce the comparisons, obtain the original repositories and set the parent directory containing their `bip/`, `FAIR/`, and `WVM/` Python folders on `PYTHONPATH`. Do not put downloaded sources inside this repository.

Suggested layout outside this project:

```text
external/
    bip/
    FAIR/
    WVM/
```

If the downloaded package folders have these exact names, use:

```bash
export PYTHONPATH="/absolute/path/to/external:/absolute/path/to/external/FAIR:$PYTHONPATH"
```

For BIP, the comparison code calls `bip.core_utils.pi_analytical` and `bip.utils.get_all_zs`. Source: [BayesianInvariantPrediction](https://github.com/LuhuanWu/BayesianInvariantPrediction). The uploaded ZIP only contains `bip/` and no upstream commit reference; verify the exact revision before full reproduction.

For FAIR, the comparison code uses `FAIR.methods.fair_algo.FairMLP`, `FairGumbelAlgo`, and `FAIR.utils.aos_default_hyper_params`. Source: [FAIR](https://github.com/wmyw96/FAIR). Upstream `FAIR.methods.fair_algo` imports `data.utils`, so the FAIR root must also be on `PYTHONPATH`. Use the original upstream repository's installation instructions rather than copying individual files into this one.

For WVM, the comparison code uses the **Python translation** at `WVM.py_implementation.wasserstein_variance` (`DirectWassersteinVariance` and `WV_screening`). The Python implementation bundled in the supplied archive is not the Julia code. The source ZIP identifies its origin as `astrzalk/WVM_reproducibility`. Place the folder containing `WVM/` on `PYTHONPATH`.

Other baselines:

- `causalicp` must be separately installed for linear ICP (import name `causalicp`).
- Nonlinear ICP uses R packages `nonlinearICP` and `CondIndTests` (see `docs/r.md`).
- EILLS, Causal Dantzig, NegDRO, CoCo, and IRM: the implementations in `baselines/` are maintained locally rather than imported from upstream packages. The EILLS implementation is adapted from [EILLS](https://github.com/wmyw96/EILLS); verify upstream redistribution terms before publicly publishing that specific adapted source file. Other baseline algorithms should be checked against the versions used for the paper.

The `baselines` Python package is a sibling of `invariant_prediction` under `src/`, not a submodule of the proposed methodology. Imports such as `from baselines.eills import fit_eills` are supported after `pip install -e .`.

The core Task 1 and Task 2 coordinate-descent methods do **not** require any of these external packages. Exact MIP requires Gurobi and a suitable license. The supplied CSVs let you reproduce existing figures without running baselines.
