# R plotting and nonlinear ICP

## R plotting

Each experiment includes R scripts for generating figures from the CSV results in `results/`. The figures are saved in the corresponding `figures/` folder.

Install the R plotting packages:

```r
install.packages(c("tidyverse","dplyr","ggplot2","patchwork","cowplot","latex2exp","readr"))
```

For example, to reproduce Figure 7 from the provided results, run the following command from the repository root:

```bash
Rscript experiments/fig07_baselines/plot.R
```

See the main README for the plotting scripts corresponding to each figure.

## Nonlinear ICP

The nonlinear ICP implementation is provided in `experiments/fig11_nonlinear/run_nonlinear_icp.R`. It requires the R packages `nonlinearICP` and `CondIndTests`, which can be installed following their package instructions.

To generate simulation data and run nonlinear ICP for one trial:

```bash
python experiments/fig11_nonlinear/generate_icp_data.py --n 100 --trial 2 --seed 42
Rscript experiments/fig11_nonlinear/run_nonlinear_icp.R 100 2
```

Here, `n` denotes the sample size per environment, and `trial` is the zero-based trial index. The Python script generates two files in `experiments/fig11_nonlinear/data/`:

- `data_100_2.csv`: covariates, response, and environment labels.
- `true_100_2.csv`: true causal parent indices.

The R script saves results as an RDS file in `nonlinear_results/`. For multiple trials, run the corresponding sample sizes and trial indices.

The repository includes the aggregated results `RS_R.csv` and `RS_py.csv` for Figure 11. To generate the figure from these results, run:

```bash
Rscript experiments/fig11_nonlinear/plot.R
```
