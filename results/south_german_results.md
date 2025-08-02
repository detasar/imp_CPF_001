# South German Credit — Experiment Summary

## Dataset

The South German Credit dataset (corrected version) comprises
1 000 German loan applicants with 20 attributes and a binary label
`kredit` (1 = good credit, 0 = bad credit) \citep{southgermandataset}.
We derive a sensitive attribute `S` from the `famges` (personal
status and sex) variable: `S=1` indicates female borrowers (code 4),
and `S=0` male borrowers.  The target `Y` equals the corrected
`kredit` field.  Each applicant’s categorical features are one-hot
encoded.

## Methods

The experiment follows the protocol described in Section~\ref{sec:empirical}:

- **Baseline** logistic regression (accuracy and fairness metrics).
- **Marginal CP**: split CP with global `alpha=0.10`.
- **Mondrian CP**: group-conditional CP on `S` with `alpha=0.10`.
- **Tuned Mondrian**: grid search over `(alpha0, alpha1) \in [0.05,0.20]`
  to reduce acceptance-rate disparity while maintaining ≥80 % coverage per
  group.

## Results

| Method | Coverage | Accuracy (accepted) | Acceptance rate | DP gap | EO gap | Remarks |
|-------|---------|----------------------|-----------------|-------|-------|---------|
| **Baseline** | – | **74.0 %** | 100 % | 0.126 | 0.042 | Strong gender disparity; small dataset |
| **Marginal CP** | 88.5 % | 83.2 % | 68.5 % | 0.156 | 0.073 | Coverage improved, but DP gap remains |
| **Mondrian CP** | 89.5 % | 83.1 % | 62.0 % | 0.156 | 0.073 | Equal coverage but large acceptance gap |
| **Tuned Mondrian** | **81.0 %** | 77.2 % | 83.5 % | 0.199 | 0.120 | Balanced acceptance; coverage drops; fairness still limited |

### Interpretation

The baseline model shows substantial demographic parity (DP) and
equalised odds (EO) gaps: female applicants are much more likely to
receive negative decisions.  Marginal CP increases coverage and
improves accuracy on accepted cases, but fairness gaps remain.  The
default Mondrian CP enforces equal coverage for men and women, but
does so by deferring many more women, exacerbating acceptance
disparities.  Tuning the miscoverage levels per group increases
acceptance for women and equalises acceptance rates; however, DP and
EO gaps remain high and overall coverage decreases.  These results
highlight the difficulty of achieving both fairness and high
performance on small, imbalanced datasets: group-specific calibration
alone may not suffice without additional bias-mitigation techniques or
larger sample sizes.
