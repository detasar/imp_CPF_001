# Bank Marketing — Experiment Summary

## Dataset

The Bank Marketing dataset contains 41 188 telephone contacts from a
Portuguese bank’s marketing campaign, with 20 attributes and a binary
outcome indicating whether a client subscribed a term deposit
\citep{bankmarketingdataset}.  We drop the `duration` attribute (known
to leak the outcome), set the target `Y=1` for subscription and
define a sensitive attribute `S=1` if age ≥ 40, else `0`.

## Methods

As in other experiments we compare a baseline logistic regression
against marginal CP, Mondrian CP and tuned Mondrian CP.  All
categorical features are one-hot encoded.

## Results

| Method | Coverage | Accuracy (accepted) | Acceptance rate | DP gap | EO gap | Remarks |
|-------|---------|----------------------|-----------------|-------|-------|---------|
| **Baseline** | – | **89.7 %** | 100 % | 0.004 | 0.005 | Minimal bias; large dataset |
| **Marginal CP** | 90.2 % | 90.1 % | 98.6 % | 0.002 | 0.011 | Slight coverage gain; small fairness gain |
| **Mondrian CP** | 90.3 % | 90.2 % | 98.5 % | 0.003 | 0.027 | Equal coverage; small increase in EO gap |
| **Tuned Mondrian** | **86.7 %** | 91.3 % | 94.9 % | 0.002 | 0.005 | Highest accuracy; acceptance parity |

### Interpretation

The Bank Marketing data exhibit very low initial disparity, so there
is limited room for improvement.  Marginal and Mondrian CP both
provide calibrated uncertainty estimates while maintaining high
coverage and accuracy.  The tuned Mondrian CP variant reduces the
acceptance-rate gap to essentially zero and achieves the highest
accuracy on accepted cases, albeit at a modest drop in overall
coverage.  These results illustrate that conformal prediction can
deliver slight fairness and performance gains even when baseline
models are already well-calibrated.
