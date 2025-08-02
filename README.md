# imp_CPF_001
# Fairness‐Aware Conformal Prediction Experiments

This repository accompanies the experiments described in the paper
*Enhancing Fairness in Artificial Intelligence Systems for Banking Applications: A Conformal
Prediction Perspective*.  It contains code to reproduce the analysis
presented in the empirical section of the paper, along with brief
interpretations of the resulting fairness and performance metrics.

## Repository structure

imp_CPF_001/

├── data/ # place raw CSV/ASC dataset files here

├── results/ # experimental summaries for each dataset

├── src/ # Python code for experiments and utilities

│ ├── common.py # fairness metrics and conformal prediction helpers

│ ├── bank_churn_experiment.py

│ ├── south_german_experiment.py

│ ├── south_german_oversampled_experiment.py

│ ├── bank_marketing_experiment.py

│ └── german_credit_experiment.py

└── README.md # this file


### Data

The actual dataset files are **not** included in this repository due to
size and licensing constraints.  To run the experiments, download
each dataset from its respective source and place the file in the
`data/` directory:

| Dataset | Source | Expected filename |
| --- | --- | --- |
| **Bank Customer Churn** | Kaggle – `bank_customer_churn.csv` | `bank_customer_churn.csv` |
| **South German Credit** (corrected) | UCI Machine Learning Repository | `south_german_credit.asc` |
| **Bank Marketing** | UCI Machine Learning Repository | `bank-full.csv` |
| **German Credit (Statlog)** | UCI ML Repository | `german.data` |

For the South German credit experiments we also create an oversampled
version of the training set on the fly, so no additional data file is
needed.  The Bank Marketing dataset uses a semicolon (`;`) as the
separator; please download the `bank-full.csv` file (from inside
`bank.zip`) and place it in the `data/` folder.

Once the datasets are in place, you can run any experiment with:

```bash
python src/<experiment_script>.py
```

For example, to reproduce the Bank Churn analysis:
```bash
python src/bank_churn_experiment.py
```


Each script prints a summary of baseline metrics, marginal conformal
prediction (CP), standard Mondrian CP and tuned Mondrian CP. See the
files in `results/` for a high-level interpretation of these numbers.

## Experiments
Each experiment script follows the same high‐level workflow:

1. Load and preprocess the data: derive a binary target Y,
sensitive attribute S (gender or age), and one‐hot encode
categorical features.

2. Train a baseline classifier: a logistic regression with
standardised inputs.

3. Compute baseline fairness metrics: demographic parity gap,
equalised odds (TPR/FPR), equal opportunity and false negative rate
differences.

4. Apply conformal prediction:

  4.1 Marginal CP with a fixed miscoverage level (α=0.10).

  4.2 Mondrian CP with the same α per group.

  4.3 Tuned Mondrian CP found by grid search over group‐specific α
  values to minimise acceptance-rate disparities while respecting
  coverage constraints.

The logic for fairness metrics and conformal prediction is encapsulated
in `src/common.py` for reuse across datasets.

## Results
Interpretations of the results are provided in the `results/` folder
for each dataset. These Markdown files summarise the key numbers
printed by the scripts and relate them to the hypotheses in the
paper.



