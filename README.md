# Convex Mixed-Integer Programming for Invariant Predictions from Multi-Environment Data

We consider two related problems for learning causal predictors from multiple environments using the **least-squares coefficient-and-risk (LSCR) invariance principle**:

- **Task 1: invariant prediction:** estimate a sparse invariant model, with convex mixed-integer optimization or a coordinate-descent heuristic (`task1.py`).
- **Task 2: intersection of invariant sets:** identify predictors shared across invariant sets through constrained invariance optimization, candidate supports, sample splitting, and multiplier bootstrap (`task2.py`, `bootstrap.py`).

The package also provides nonlinear/group basis selection, a hidden confounding extension, and synthetic data generators. Independent baseline implementations and adapters live in the separate `baselines` package.

## Install

Python 3.10+ is required. In the repository root:

```bash
python -m pip install -e .
```

This installs the core methods. To run exact MIP, install Gurobi and ensure that your Gurobi license supports your model size:

```bash
python -m pip install -e '.[mip]'
```

Several experiments require PyTorch (`pip install -e '.[torch]'`) and separately downloaded baseline methods; see [baseline instructions](docs/baselines.md). R is only needed for plotting or nonlinear ICP; see [R instructions](docs/r.md).

## Quick example

```python
from invariant_prediction import cd_msetype,precompute_stat
from invariant_prediction.data import generate_synthetic_data_soft_intervention

S_true,beta_true,X_list,y_list,n_list=generate_synthetic_data_soft_intervention(
    n=100,p=10,E=3,p_act=.2,lb=.1,ub=.2,lb_star=.5,ub_star=1,
    sigma_x_min=.5,sigma_x_max=1,sigma_y=.5,p_star_min=2,p_star_max=2,
    p_resample=.1
)
S_hat,B,obj=cd_msetype(precompute_stat(X_list,y_list),lam=.01,gamma1=5,rho=1)
print(S_hat)
```

The coordinate-descent example does not require Gurobi or external baselines. See `tests/` for small Task 1, Task 2, and nonlinear smoke checks.

## Paper experiments and figures

The committed CSV results can be plotted without rerunning simulations. Each experiment folder has a `run.py` that generates new result CSVs, one or more R plotting scripts, the original manuscript CSVs in `results/`, and the supplied figure PDF in `figures/`.

| Figure(s) | Experiment folder | Appendix | Plotting |
| --- | --- | --- | --- |
| 2 | `fig02_solver_comparison` | G.1 | `plot.R` |
| 3 | `fig03_identification` | G.2 | `plot.R` |
| 4, 5, 8 | `fig04_05_08_example` | G.5 | `plot_fig05.R`, `plot_fig08.R` |
| 6 | `fig06_sample_size_tuning` | G.3 | `plot.R` (sources `plot_left_middle.R`) |
| 7 | `fig07_baselines` | G.4 | `plot.R` |
| 9 | `fig09_testing` | G.6 | `plot.R` |
| 10 | `fig10_causal_chamber` | G.7 | `plot.R` |
| 11 | `fig11_nonlinear` | G.8 | `plot.R` and `run_nonlinear_icp.R` |
| 12 | `fig12_confounding` | G.9 | `plot.R` |

Example: reproduce Figure 7 directly from the **provided results**:

```bash
Rscript experiments/fig07_baselines/plot.R
```

Generate new Figure 2 simulation results after installing Gurobi:

```bash
python experiments/fig02_solver_comparison/run.py --trials 2 --seed 42
```

New simulations write to `results/RS_generated.csv`, rather than overwriting original paper results. `--output` specifies another destination. Run `python experiments/<folder>/run.py --help` for the common command-line options. Figure 6 also has `run_tuning.py` for the right-hand tuning panel.


## Project layout

```text
src/invariant_prediction/    Our methods, extensions, and synthetic data
src/baselines/               Independent baseline implementations and adapters
experiments/                 Figure-by-figure simulation scripts, original CSVs, R plots
notebooks/                   Original source notebook for comparison
paper/                       Reference manuscript
tests/                       Lightweight method and packaging tests
docs/                        External dependency, R, and reproducibility notes
```

For the nonlinear ICP R experiment, generate the required data files first:

```bash
python experiments/fig11_nonlinear/generate_icp_data.py --n 100 --trial 2 --seed 42
Rscript experiments/fig11_nonlinear/run_nonlinear_icp.R 100 2
```

The generated `data_*.csv` and `true_*.csv` are temporary simulation inputs, not the published figure results, and are intentionally not bundled. Figure 11 can still be plotted directly from its bundled result CSVs.

## Third-party software, licensing, and attribution

BIP, FAIR, and WVM are **external optional dependencies** and their source code is not included. 
