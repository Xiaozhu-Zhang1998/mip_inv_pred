# External baseline methods

BIP and FAIR are optional external dependencies, with adapters in `src/baselines/`. Their original packages must be installed separately to rerun those comparisons. WVM is implemented locally in `src/baselines/wvm.py` and requires no separate installation. For BIP and FAIR, place the relevant external package directories on `PYTHONPATH`.

Suggested layout outside this project:

```text
external/
    bip/
    FAIR/
```

If the downloaded package folders have these exact names, use:

```bash
export PYTHONPATH="/absolute/path/to/external:/absolute/path/to/external/FAIR:$PYTHONPATH"
```

For BIP, the comparison code calls `bip.core_utils.pi_analytical` and `bip.utils.get_all_zs`. Source: [BayesianInvariantPrediction](https://github.com/LuhuanWu/BayesianInvariantPrediction). The uploaded ZIP only contains `bip/` and no upstream commit reference; verify the exact revision before full reproduction.

For FAIR, the comparison code uses `FAIR.methods.fair_algo.FairMLP`, `FairGumbelAlgo`, and `FAIR.utils.aos_default_hyper_params`. Source: [FAIR](https://github.com/wmyw96/FAIR). Upstream `FAIR.methods.fair_algo` imports `data.utils`, so the FAIR root must also be on `PYTHONPATH`. Use the original upstream repository's installation instructions rather than copying individual files into this one.

For WVM, the Python implementation is included in `src/baselines/wvm.py`, providing `DirectWassersteinVariance` and `WV_screening`. It is a translation based on the original [WVM Julia implementation](https://github.com/astrzalk/WVM_reproducibility). The original Julia project is MIT licensed (see `LICENSES/WVM-MIT`). The origin of this Python translation has not been verified; it should not be described as an official release by the original authors.

Other baselines:

- `causalicp` must be separately installed for linear ICP (import name `causalicp`).
- Nonlinear ICP uses R packages `nonlinearICP` and `CondIndTests` (see `docs/r.md`).
- EILLS, Causal Dantzig, NegDRO, CoCo, and IRM: the implementations in `baselines/` are maintained locally rather than imported from upstream packages. The EILLS implementation is adapted from [EILLS](https://github.com/wmyw96/EILLS).

The `baselines` Python package is a sibling of `invariant_prediction` under `src/`, not a submodule of the proposed methodology. Imports such as `from baselines.eills import fit_eills` are supported after `pip install -e .`.

The core Task 1 and Task 2 coordinate descent methods do **not** require BIP, FAIR, or other optional baseline packages. Exact MIP requires Gurobi and a suitable license. The supplied CSVs let you reproduce existing figures without running baselines.
