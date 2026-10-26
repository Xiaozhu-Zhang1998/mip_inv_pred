# Reproducibility notes and known discrepancies

This repository provides independently executable experiment scripts and the CSV results used for the figures. Some script settings have not yet been reconciled with the experimental specifications in Appendix G. **Do not assume that a new simulation run exactly reproduces the published protocol.** The documented experimental specifications should guide a future reconciliation.

Known differences and execution fixes to review:

- Figure 2: the script includes dimensions up to 60; Appendix G.1 ends at 50.
- Figure 3: the script has 5,000 trial iterations; the published experiment reports 50 summary replications with repeated datasets.
- Figure 7: the script uses 10 trials and fixed tuning parameters; Appendix G.4 specifies 200 replications, LOEO tuning, and a different dimension regime.
- Figure 9: the script uses 100 trials rather than 200. The experiment's output rows use the number of environments `E`; this resolves a variable mismatch in the earlier simulation code.
- Figure 10: the experiment constructs `sampled_data` but fits the unsampled `data`. This does not yet implement the observation-resampling step described in Appendix G.7.
- Figure 11: `generate_icp_data.py` writes both the simulated data and true causal support required by `run_nonlinear_icp.R`. Aggregated nonlinear ICP results are provided but not regenerated automatically.
- Figure 6: the sample-size experiment explicitly initializes its results collection before appending trial results.

The supplied figure results remain in each experiment's `results/` directory (except that `baseline_p_2.csv` was renamed to `baseline_p.csv`). Newly run experiments save to `results/RS_generated.csv` (or `RS_reg_generated.csv`) by default, so existing results are not overwritten unless a matching path is explicitly supplied with `--output`.

Some experiments require optional dependencies, a licensed Gurobi installation, or separate upstream baseline packages. Tests cover core algorithms on small data only. The Python scripts have been syntax checked; the complete experiments and R plots have not been validated end to end.

The large `causal_chamber_trials.pkl` cache and redundant nested chamber ZIP are excluded. The Figure 10 `data/` folder instead contains the environment CSV files used in that experiment. Review the upstream data redistribution terms before publishing the repository.
