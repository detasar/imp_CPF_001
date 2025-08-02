# German Credit (Statlog) — Experiment Summary

## Dataset

The Statlog German Credit dataset includes 1 000 customers with 20
mixed-type attributes and a binary credit classification (1 = good
credit, 2 = bad credit) \citep{statlogdataset}.  We convert the
target to `Y=1` for bad credit, `0` for good.  A sensitive attribute
`S` is derived from age: `S=1` if age ≥ the median age, else `0`.
Numeric columns are kept numeric and all categorical columns are
one-hot encoded.

## Methods

We reuse the same experimental pipeline as in other datasets: baseline
logistic regression, marginal CP, Mondrian CP and a tuned
group-specific CP.  The grid search for tuning aims to equalise
acceptance rates while maintaining at least 80 % coverage per group.

## Results

| Method | Coverage | Accuracy (accepted) | Acceptance rate | DP gap | EO gap | Remarks |
|-------|---------|----------------------|-----------------|-------|-------|---------|
| **Baseline** | – | **75.0 %** | 100 % | 0.250 | 0.225 | Significant age disparity |
| **Marginal CP** | 90.5 % | 85.0 % | 63.5 % | 0.162 | 0.129 | Coverage ↑, disparity ↓ |
| **Mondrian CP** | 89.0 % | 83.5 % | 66.5 % | 0.123 | 0.159 | Equal coverage; acceptance gap persists |
| **Tuned Mondrian** | **93.5 %** | 88.4 % | 56.0 % | 0.174 | 0.221 | Acceptance parity; EO gap remains high |

### Interpretation

The baseline classifier on the German Credit data displays a large
demographic parity gap and unequal error rates between younger and
older applicants.  Marginal CP improves both coverage and fairness but
requires rejecting about a third of applications.  Mondrian CP
equalises coverage across age groups but increases the acceptance
disparity.  Tuning the group-specific miscoverage levels reduces the
acceptance gap but results in only marginal fairness improvement and
even increases the EO gap.  Thus, while conformal methods offer
significant reliability gains, achieving both fairness and high
performance on this dataset remains challenging.
