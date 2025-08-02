# Bank Customer Churn — Experiment Summary

## Dataset

The Bank Customer Churn dataset contains 10 000 customer records with
demographic and account information.  The sensitive attribute is
`Gender` (1 = female, 0 = male) and the target `Y` is whether the
customer churned (`Exited`).  See
<https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling>.

## Methods

- **Baseline**: logistic regression on standardised features.  Computes
  accuracy and fairness metrics (demographic parity difference,
  equalised odds gaps, equal opportunity, FNR difference).
- **Marginal CP**: split conformal prediction with a global
  miscoverage level `alpha=0.10`.
- **Mondrian CP**: group‐conditional calibration on `Gender` with
  `alpha=0.10`.
- **Tuned Mondrian**: grid search over `(alpha0, alpha1)` in
  `[0.05, 0.20]` to minimise acceptance-rate gap while enforcing
  ≥80 % coverage per group.

## Results

| Method | Coverage | Accuracy on accepted | Acceptance rate | DP gap | EO gap | Remarks |
|-------|----------|----------------------|-----------------|-------|-------|---------|
| **Baseline** | – | **81.4 %** | 100 % | 0.009 | 0.022 | Naïve model; small gender disparity |
| **Marginal CP** | 89.7 % | 86.9 % | 78.5 % | **0.002** | **0.006** | Coverage ↑, fairness improved |
| **Mondrian CP** | 89.9 % | 87.2 % | 79.5 % | 0.016 | 0.078 | Coverage parity, but female acceptance lower |
| **Tuned Mondrian** | **90.1 %** | 87.1 % | 77.0 % | **0.001** | **0.008** | Nearly perfect acceptance parity and very small fairness gaps |

### Interpretation

The baseline classifier already exhibits relatively minor disparities
between genders.  Marginal CP improves overall reliability and
slightly reduces fairness gaps while declining to make a definitive
prediction on about 21 % of customers.  The default Mondrian CP
achieves equal coverage for men and women, but it does so by deferring
more female customers, increasing acceptance disparity.  Tuning
`alpha` separately for each gender eliminates this disparity and
preserves high coverage and accuracy.  Overall, the tuned
Mondrian CP demonstrates that group-specific calibration can offer
fairer decisions without sacrificing performance.
