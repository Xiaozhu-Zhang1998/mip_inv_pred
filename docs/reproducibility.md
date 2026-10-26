# Reproducibility notes and known discrepancies

The first refactor concentrates on packaging rather than reconciling differences between the notebook and the paper. Experiment scripts preserve the notebook's parameter choices and loops, except for basic wiring fixes and optional trial/seed/output CLI arguments. **Appendix G of `paper/manuscript.pdf` takes precedence for the eventual paper-accurate simulation revision.** Do not assume rerunning the current scripts reproduces the precise manuscript protocol.

Examples to reconcile later:

- Figure 2: notebook dimension grid goes up to 60; the manuscript Appendix G.1 goes up to 50.
- Figure 3: notebook includes 5,000 trial iterations; the paper reports 50 summary replications with repeated datasets.
- Figure 7: notebook has 10 trial iterations and fixed tuning; the manuscript specifies 200 replications with LOEO tuning and a different dimension regime.
- Figure 9: notebook has 100 trials rather than the paper's 200. The original notebook reassigns `n` and uses an undefined `n_` for result rows; the script repairs these execution errors by using the number of environments `E`.
- Figure 10: the notebook constructs `sampled_data` but still fits the unsampled `data`. Preserved for now, as requested; rerunning does not yet implement every aspect of Appendix G.7.
- Figure 11: the original Python cell did not export `true_<n>_<trial>.csv` despite the nonlinear ICP R dependency. The script now exports the true support as well as the data, while retaining the supplied example CSVs. The supplied R aggregate is not regenerated automatically.
- Figure 6: the notebook omitted initialization of `RS` in its sample-size cell. Added explicitly so the script executes independently.

Original results are preserved **verbatim** in `results/` (except renaming `baseline_p_2.csv` to `baseline_p.csv`). Newly run experiments save to `results/RS_generated.csv` (or `RS_reg_generated.csv`) by default and never replace the paper files unless you explicitly pass `--output`.

Some experiments are computationally expensive and require optional dependencies, licensed Gurobi, and/or special upstream packages. Tests cover core algorithms on small data only. All scripts were syntax checked, but the complete paper experiments were not rerun.

The 477 MB `causal_chamber_trials.pkl` cache and the redundant nested original chamber ZIP are not included. Instead, the 20 environment CSVs directly referenced by the notebook are kept in the Figure 10 `data/` directory. Review upstream data redistribution terms before making the repository public.
