# R plotting and nonlinear ICP

Every experiment contains its original ggplot2 script (`plot.R` or a figure-specific filename), updated to read the bundled CSV from `results/`. Run it from anywhere using `Rscript experiments/<folder>/plot.R`; the script changes to its own directory and writes `figures/plot*.pdf`.

Install the CRAN plotting dependencies as needed:

```r
install.packages(c("tidyverse","dplyr","ggplot2","patchwork","cowplot","latex2exp","readr"))
```

Nonlinear ICP is a **separate R computation**, not a plotting script. Its code was moved from the SLURM-based `R_part.R` to `experiments/fig11_nonlinear/run_nonlinear_icp.R`. It no longer uses `util_multi_index.r` or SLURM array indices. Install `nonlinearICP` and `CondIndTests` according to their upstream package instructions. To generate input data and run one example:

```bash
python experiments/fig11_nonlinear/generate_icp_data.py --n 100 --trial 2 --seed 42
Rscript experiments/fig11_nonlinear/run_nonlinear_icp.R 100 2
```

The arguments to the R runner are the number of observations per environment and a zero-based trial index. The Python generator creates `data/data_100_2.csv` (covariates, response, environment) and `data/true_100_2.csv` (true causal parent indices). These temporary files are not required for plotting Figure 11 and are not shipped in the repository. The R runner produces an RDS file in `nonlinear_results/`; the paper's aggregated `results/RS_R.csv` is already provided. For a full nonlinear ICP simulation, generate the inputs for every sample size and trial before running the corresponding R commands. This example does not regenerate the full paper figure.

To plot the original paper Figure 11, run:

```bash
Rscript experiments/fig11_nonlinear/plot.R
```

The R executable and R packages were not present in the build environment, so R execution remains unverified. The original plotting logic has otherwise been retained.
