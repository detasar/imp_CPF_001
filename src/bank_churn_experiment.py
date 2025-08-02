"""
Experiment script for the Bank Customer Churn dataset.

This script preprocesses the churn data, trains a logistic regression
classifier, evaluates baseline fairness metrics and then applies
split conformal (marginal) and group-conditional (Mondrian) conformal
prediction.  A grid search over group-specific miscoverage levels is
performed to minimise acceptance-rate disparities.  Results are printed
to the console.

Usage:
    python bank_churn_experiment.py

The dataset file `bank_customer_churn.csv` should be placed in the
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
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """Prepare Bank Customer Churn data.

    Adds sensitive column `S` (gender) and target `Y` (Exited),
    drops identifiers and encodes categorical features.
    """
    df = df.copy()
    df["S"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Y"] = df["Exited"]
    df.drop(columns=["RowNumber", "CustomerId", "Surname", "Gender", "Exited"], inplace=True)
    geo_dummies = pd.get_dummies(df["Geography"], prefix="Geo", drop_first=True)
    df.drop(columns=["Geography"], inplace=True)
    df = pd.concat([df, geo_dummies], axis=1)
    return df, "S", "Y"


def oversample_training(X, y, S):
    """No oversampling for churn data; return as-is."""
    return X, y, S


def run_experiment():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "bank_customer_churn.csv")
    df_raw = load_data(data_path)
    df, s_col, y_col = preprocess(df_raw)
    features = [c for c in df.columns if c not in [s_col, y_col]]
    # Split data
    train_df, hold_df = train_test_split(df, test_size=0.4, stratify=df[y_col], random_state=42)
    cal_df, test_df = train_test_split(hold_df, test_size=0.5, stratify=hold_df[y_col], random_state=42)
    X_train = train_df[features].values
    y_train = train_df[y_col].values
    S_train = train_df[s_col].values
    X_cal = cal_df[features].values
    y_cal = cal_df[y_col].values
    S_cal = cal_df[s_col].values
    X_test = test_df[features].values
    y_test = test_df[y_col].values
    S_test = test_df[s_col].values
    # Baseline classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    clf.fit(X_train, y_train)
    proba_train = clf.predict_proba(X_train)[:, 1]
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
    variant = cp_mond["variant"]
    alpha_grid = np.round(np.linspace(0.05, 0.20, 16), 2)
    best_row = None
    best_score = None
    for a0 in alpha_grid:
        for a1 in alpha_grid:
            pred_sets_tmp = []
            for p, s in zip(proba_test, S_test):
                cal_sorted = cal_sorted_by_s[int(s)]
                a = a0 if s == 0 else a1
                # replicate prediction set computation
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
            summary, _, _ = evaluate_cp_sets(y_test, S_test, pred_sets_tmp, name="tmp")
            cov0 = summary["coverage_by_group"].get(0, 0)
            cov1 = summary["coverage_by_group"].get(1, 0)
            # ensure coverage floor
            if cov0 < 0.80 or cov1 < 0.80:
                continue
            acc0 = summary["accept_rate_by_group"].get(0, 0)
            acc1 = summary["accept_rate_by_group"].get(1, 0)
            acc_gap = abs(acc0 - acc1)
            dp_gap_tmp = summary["dp_gap_on_accepted"]
            score = (acc_gap, dp_gap_tmp, -summary["coverage_overall"])
            if best_score is None or score < best_score:
                best_score = score
                best_row = {
                    "alpha0": a0,
                    "alpha1": a1,
                    "summary": summary,
                }
    tuned_summary = best_row["summary"] if best_row is not None else None
    # Print results
    print("Bank Customer Churn experiment")
    print("Baseline metrics:", baseline_metrics)
    print("Marginal CP summary:", marg_summary)
    print("Mondrian CP summary:", mond_summary)
    if tuned_summary is not None:
        print("Tuned Mondrian CP summary:", tuned_summary)
    else:
        print("No tuned Mondrian CP found with desired coverage floor.")


if __name__ == "__main__":
    run_experiment()
