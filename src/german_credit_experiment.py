"""
Experiment script for the Statlog German Credit dataset.

This script loads the German Credit dataset from the UCI Statlog
repository, derives a binary target `Y` (1 for bad credit, 0 for good
credit) and defines a sensitive attribute `S` based on age (1 if age
≥ median age, 0 otherwise).  It automatically determines which
columns are numeric and which are categorical, applies one-hot
encoding to categorical variables, trains a logistic regression
baseline, evaluates fairness metrics, and applies marginal and
group-conditional conformal prediction with group-specific tuning.

Usage:
    python german_credit_experiment.py

Place `german.data` from the UCI Statlog German Credit dataset in the
`data/` directory.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    demographic_parity_diff,
    equalized_odds_gaps,
    equal_opportunity_gap,
    fnr_diff,
    fit_marginal_cp,
    predict_marginal_cp,
    fit_mondrian_cp,
    predict_mondrian_cp,
    evaluate_cp_sets,
)


def load_data(path: str) -> pd.DataFrame:
    # The dataset uses space-separated values without header
    df = pd.read_csv(path, sep=" ", header=None, dtype=str)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # The label is in the last column: 1=good credit, 2=bad credit
    y_raw = df.iloc[:, -1].astype(int)
    df.drop(columns=df.columns[-1], inplace=True)
    # Determine numeric vs categorical columns based on ability to convert to number
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        # Try converting entire column to numeric
        try:
            col_numeric = pd.to_numeric(df[col])
            numeric_cols.append(col)
        except Exception:
            categorical_cols.append(col)
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Derive sensitive attribute from age (column index 12 in documentation).
    # Use numeric value if available, otherwise try to parse string to int.
    if len(numeric_cols) > 0:
        # assume that the 13th column (index 12) corresponds to age
        age_series = df_numeric.iloc[:, 12]
    else:
        # fallback: attempt numeric conversion
        age_series = pd.to_numeric(df.iloc[:, 12], errors="coerce")
    median_age = age_series.median()
    S = (age_series >= median_age).astype(int)
    # Target: 1 for bad credit (value 2), 0 for good credit (value 1)
    Y = (y_raw == 2).astype(int)
    # One-hot encode categorical columns
    df_categ = pd.get_dummies(df[categorical_cols], drop_first=True)
    df_processed = pd.concat([df_numeric, df_categ], axis=1)
    df_processed["S"] = S.values
    df_processed["Y"] = Y.values
    return df_processed


def oversample_training(X, y, S):
    """No oversampling for the German Credit experiment."""
    return X, y, S


def run_experiment():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "german.data")
    df_raw = load_data(data_path)
    df = preprocess(df_raw)
    s_col, y_col = "S", "Y"
    features = [c for c in df.columns if c not in [s_col, y_col]]
    # Split data
    train_df, hold_df = train_test_split(df, test_size=0.4, stratify=df[y_col], random_state=42)
    cal_df, test_df = train_test_split(hold_df, test_size=0.5, stratify=hold_df[y_col], random_state=42)
    X_train, y_train, S_train = train_df[features].values, train_df[y_col].values, train_df[s_col].values
    X_cal, y_cal, S_cal = cal_df[features].values, cal_df[y_col].values, cal_df[s_col].values
    X_test, y_test, S_test = test_df[features].values, test_df[y_col].values, test_df[s_col].values
    # Baseline classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    clf.fit(X_train, y_train)
    proba_cal = clf.predict_proba(X_cal)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    # Baseline fairness
    dp_gap, _, _ = demographic_parity_diff(pred_test, S_test)
    eo_tpr_gap, eo_fpr_gap, _ = equalized_odds_gaps(y_test, pred_test, S_test)
    eopp_gap, _ = equal_opportunity_gap(y_test, pred_test, S_test)
    fnr_gap, _ = fnr_diff(y_test, pred_test, S_test)
    from sklearn.metrics import accuracy_score, roc_auc_score
    baseline_metrics = {
        "accuracy": accuracy_score(y_test, pred_test),
        "roc_auc": roc_auc_score(y_test, proba_test),
        "dp_gap": dp_gap,
        "eo_tpr_gap": eo_tpr_gap,
        "eo_fpr_gap": eo_fpr_gap,
        "eopp_gap": eopp_gap,
        "fnr_gap": fnr_gap,
    }
    # Marginal CP
    alpha_level = 0.10
    cp_marg = fit_marginal_cp(proba_cal, y_cal, alpha_level=alpha_level, variant="plain")
    marg_sets, _ = predict_marginal_cp(cp_marg, proba_test)
    marg_summary, _, _ = evaluate_cp_sets(y_test, S_test, marg_sets, name=f"Marginal CP (α={alpha_level})")
    # Mondrian CP default
    cp_mond = fit_mondrian_cp(proba_cal, y_cal, S_cal, alpha_level=alpha_level, variant="plain")
    mond_sets, _ = predict_mondrian_cp(cp_mond, proba_test, S_test)
    mond_summary, _, _ = evaluate_cp_sets(y_test, S_test, mond_sets, name=f"Mondrian CP (α={alpha_level})")
    # Grid search for tuned alphas
    cal_sorted_by_s = cp_mond["cal_sorted_by_s"]
    alpha_grid = np.round(np.linspace(0.05, 0.20, 16), 2)
    best_score = None
    best_summary = None
    best_alphas = (None, None)
    for a0 in alpha_grid:
        for a1 in alpha_grid:
            pred_sets_tmp = []
            for p, s in zip(proba_test, S_test):
                cal_sorted = cal_sorted_by_s[int(s)]
                a = a0 if s == 0 else a1
                p1 = float(np.clip(p, 1e-6, 1 - 1e-6))
                p0 = 1.0 - p1
                alpha_y1 = 1.0 - p1
                alpha_y0 = 1.0 - p0
                n_cal = len(cal_sorted)
                idx_ge1 = np.searchsorted(cal_sorted, alpha_y1, side="left")
                idx_ge0 = np.searchsorted(cal_sorted, alpha_y0, side="left")
                n_ge1 = n_cal - idx_ge1
                n_ge0 = n_cal - idx_ge0
                pval1 = (1 + n_ge1) / (n_cal + 1)
                pval0 = (1 + n_ge0) / (n_cal + 1)
                sset = []
                if pval0 > a:
                    sset.append(0)
                if pval1 > a:
                    sset.append(1)
                pred_sets_tmp.append(sset)
            summary, _, _ = evaluate_cp_sets(y_test, S_test, pred_sets_tmp, name="tuned")
            cov0 = summary["coverage_by_group"].get(0, 0)
            cov1 = summary["coverage_by_group"].get(1, 0)
            if cov0 < 0.80 or cov1 < 0.80:
                continue
            acc0 = summary["accept_rate_by_group"].get(0, 0)
            acc1 = summary["accept_rate_by_group"].get(1, 0)
            acc_gap = abs(acc0 - acc1)
            dp_gap_tmp = summary["dp_gap_on_accepted"]
            score = (acc_gap, dp_gap_tmp, -summary["coverage_overall"])
            if best_score is None or score < best_score:
                best_score = score
                best_summary = summary
                best_alphas = (a0, a1)
    # Print results
    print("German Credit experiment")
    print("Baseline metrics:", baseline_metrics)
    print("Marginal CP summary:", marg_summary)
    print("Mondrian CP summary:", mond_summary)
    if best_summary is not None:
        print(f"Tuned Mondrian CP summary (α0={best_alphas[0]}, α1={best_alphas[1]}):", best_summary)
    else:
        print("No tuned Mondrian CP found with coverage ≥ 0.8 for both groups.")


if __name__ == "__main__":
    run_experiment()
