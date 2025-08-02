# South German Credit — Oversampled Training — Experiment Summary

## Dataset and Oversampling

We use the same corrected South German Credit data as in the
`south_german_results.md` experiment but apply random oversampling to
the minority class (`Y=0`, bad credit) in the **training set only**.
This balances the class distribution in the classifier’s training data
while leaving calibration and test sets unchanged.

## Methods

The methods mirror those in the original South German experiment:
baseline logistic regression, marginal CP, Mondrian CP and tuned
Mondrian CP.  The oversampling is intended to mitigate class
imbalance before calibration.

## Results

| Method | Coverage | Accuracy (accepted) | Acceptance rate | DP gap | EO gap | Remarks |
|-------|---------|----------------------|-----------------|-------|-------|---------|
| **Baseline** | – | **70.5 %** | 100 % | 0.040 | 0.050 | Oversampled training improves DP gap vs. original |
| **Marginal CP** | 87.5 % | 78.6 % | 58.5 % | 0.171 | 0.050 | Coverage ↑ but fairness worsens |
| **Mondrian CP** | 88.5 % | 78.5 % | 53.5 % | 0.452 | 0.257 | Equal coverage, very large disparity |
| **Tuned Mondrian** | **90.0 %** | 77.8 % | 45.0 % | 0.196 | 0.086 | Parity in acceptance, but DP/EO gaps remain high |

### Interpretation

Balancing the training data reduces the demographic parity gap in the
baseline model, but conformal methods react unpredictably: marginal
and Mondrian CP both increase disparity relative to the oversampled
baseline.  Mondrian CP in particular defers many more bad-credit
applications, yielding a fourfold increase in DP gap.  Tuning group
miscoverage levels equalises acceptance rates but does not reduce
fairness gaps to acceptable levels.  These results suggest that naive
oversampling alone is insufficient for fairness and may even harm
downstream conformal calibration.  More sophisticated balancing or
feature design may be necessary.
