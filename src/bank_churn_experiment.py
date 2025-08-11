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
from typing import Tuple
from sklearn.model_selection import train_test_split

from common import (
    check_data_file,
    run_experiment_pipeline,
    print_experiment_results,
)


def load_data(path: str) -> pd.DataFrame:
    check_data_file(path, "Bank Customer Churn dataset")
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

    # Run the standard experiment pipeline
    results = run_experiment_pipeline(
        X_train, y_train, S_train,
        X_cal, y_cal, S_cal,
        X_test, y_test, S_test,
        experiment_name="Bank Customer Churn experiment"
    )

    # Print results
    print_experiment_results(results)


if __name__ == "__main__":
    run_experiment()
