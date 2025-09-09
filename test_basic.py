#!/usr/bin/env python3
"""
Basic tests for the fairness-aware conformal prediction codebase.

This test suite validates the core functionality and import structure
without requiring actual dataset files.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test imports
def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        import common
        print("✓ common.py imported successfully")

        from common import (
            check_data_file,
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
        from common import (
            check_data_file,
            run_experiment_pipeline,
            print_experiment_results,
        )
        print("✓ New common functions imported successfully")

        # Test experiment imports (without running them)
        import bank_churn_experiment
        import bank_marketing_experiment
        import german_credit_experiment
        import south_german_experiment
        import south_german_oversampled_experiment
        print("✓ All experiment modules imported successfully")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_fairness_metrics():
    """Test fairness metric calculations with synthetic data."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred = np.random.binomial(1, 0.4, n_samples)
    S = np.random.binomial(1, 0.5, n_samples)

    from common import (
        demographic_parity_diff,
        equalized_odds_gaps,
        equal_opportunity_gap,
        fnr_diff,
    )

    try:
        dp_gap, p0, p1 = demographic_parity_diff(y_pred, S)
        print(f"✓ Demographic parity: gap={dp_gap:.3f}, p0={p0:.3f}, p1={p1:.3f}")

        eo_tpr_gap, eo_fpr_gap, eo_detail = equalized_odds_gaps(y_true, y_pred, S)
        print(f"✓ Equalized odds: TPR gap={eo_tpr_gap:.3f}, FPR gap={eo_fpr_gap:.3f}")

        eopp_gap, eopp_detail = equal_opportunity_gap(y_true, y_pred, S)
        print(f"✓ Equal opportunity: gap={eopp_gap:.3f}")

        fnr_gap, fnr_detail = fnr_diff(y_true, y_pred, S)
        print(f"✓ FNR difference: gap={fnr_gap:.3f}")

        return True
    except Exception as e:
        print(f"✗ Fairness metrics error: {e}")
        return False


def test_experiment_pipeline():
    """Test the complete experiment pipeline with synthetic data."""
    # Create synthetic data
    np.random.seed(42)
    n_train, n_cal, n_test = 50, 30, 40
    n_features = 5

    # Generate features
    X_train = np.random.randn(n_train, n_features)
    X_cal = np.random.randn(n_cal, n_features)
    X_test = np.random.randn(n_test, n_features)

    # Generate labels and sensitive attributes
    y_train = np.random.binomial(1, 0.3, n_train)
    y_cal = np.random.binomial(1, 0.3, n_cal)
    y_test = np.random.binomial(1, 0.3, n_test)
    S_train = np.random.binomial(1, 0.5, n_train)
    S_cal = np.random.binomial(1, 0.5, n_cal)
    S_test = np.random.binomial(1, 0.5, n_test)

    from common import run_experiment_pipeline, print_experiment_results

    try:
        results = run_experiment_pipeline(
            X_train, y_train, S_train,
            X_cal, y_cal, S_cal,
            X_test, y_test, S_test,
            experiment_name="Test Experiment"
        )

        print(f"✓ Experiment pipeline ran successfully")
        print(f"✓ Baseline accuracy: {results['baseline_metrics']['accuracy']:.3f}")
        print(f"✓ Marginal CP coverage: {results['marg_summary']['coverage_overall']:.3f}")
        print(f"✓ Mondrian CP coverage: {results['mond_summary']['coverage_overall']:.3f}")
        if results['tuned_summary'] is not None:
            print(f"✓ Tuned CP found with coverage: {results['tuned_summary']['coverage_overall']:.3f}")
        else:
            print("✓ No tuned CP found (expected with small synthetic data)")

        return True
    except Exception as e:
        print(f"✗ Experiment pipeline error: {e}")
        return False


def test_conformal_prediction():
    """Test conformal prediction with synthetic data."""
    # Create synthetic data
    np.random.seed(42)
    n_cal = 50
    n_test = 30
    proba_cal = np.random.uniform(0.1, 0.9, n_cal)
    y_cal = np.random.binomial(1, proba_cal)
    S_cal = np.random.binomial(1, 0.5, n_cal)

    proba_test = np.random.uniform(0.1, 0.9, n_test)
    S_test = np.random.binomial(1, 0.5, n_test)
    y_test = np.random.binomial(1, proba_test)

    from common import (
        fit_marginal_cp,
        predict_marginal_cp,
        fit_mondrian_cp,
        predict_mondrian_cp,
        evaluate_cp_sets,
    )

    try:
        # Test marginal CP
        cp_marg = fit_marginal_cp(proba_cal, y_cal, alpha_level=0.1)
        marg_sets, marg_pvals = predict_marginal_cp(cp_marg, proba_test)
        print(f"✓ Marginal CP: {len(marg_sets)} prediction sets generated")

        # Test Mondrian CP
        cp_mond = fit_mondrian_cp(proba_cal, y_cal, S_cal, alpha_level=0.1)
        mond_sets, mond_pvals = predict_mondrian_cp(cp_mond, proba_test, S_test)
        print(f"✓ Mondrian CP: {len(mond_sets)} prediction sets generated")

        # Test evaluation
        marg_summary, accept_mask, point_pred = evaluate_cp_sets(y_test, S_test, marg_sets, "Test")
        print(f"✓ CP evaluation: coverage={marg_summary['coverage_overall']:.3f}, acceptance={marg_summary['accept_rate_overall']:.3f}")

        return True
    except Exception as e:
        print(f"✗ Conformal prediction error: {e}")
        return False


def test_error_handling():
    """Test error handling for missing data files."""
    from common import check_data_file

    try:
        # This should raise FileNotFoundError
        check_data_file("/nonexistent/file.csv", "Test dataset")
        print("✗ Error handling failed - should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"✓ Error handling works: {str(e)[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Unexpected error in error handling: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Running tests for fairness-aware conformal prediction codebase...\n")

    tests = [
        ("Module imports", test_imports),
        ("Fairness metrics", test_fairness_metrics),
        ("Experiment pipeline", test_experiment_pipeline),
        ("Conformal prediction", test_conformal_prediction),
        ("Error handling", test_error_handling),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")

    print(f"\n--- Summary ---")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {len(tests) - passed}/{len(tests)}")

    if passed == len(tests):
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)