"""
Common utilities for fairness-aware conformal prediction experiments.

This module contains fairness metrics and conformal prediction helpers
originally adapted from the user's notebook.  It provides functions to
compute demographic parity and error-rate gaps, as well as implementations
of split (marginal) and group-conditional (Mondrian) conformal
prediction for binary classification.
"""

import numpy as np
from typing import Dict, Tuple, List


def demographic_parity_diff(y_pred: np.ndarray, S: np.ndarray) -> Tuple[float, float, float]:
    """Compute demographic parity gap for binary predictions.

    Returns the absolute difference in positive prediction rates
    between the two groups S=0 and S=1, along with the individual rates.
    """
    p0 = (y_pred[S == 0] == 1).mean() if np.any(S == 0) else np.nan
    p1 = (y_pred[S == 1] == 1).mean() if np.any(S == 1) else np.nan
    return abs(p0 - p1), p0, p1


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (tp + fn + 1e-12)
    return tpr, fpr, fnr


def equalized_odds_gaps(y_true: np.ndarray, y_pred: np.ndarray, S: np.ndarray) -> Tuple[float, float, Tuple[float, float, float, float]]:
    """Compute equalized odds gaps.

    Returns absolute differences in TPR and FPR between groups, along with
    the individual group rates.
    """
    tpr0, fpr0, _ = _rates(y_true[S == 0], y_pred[S == 0])
    tpr1, fpr1, _ = _rates(y_true[S == 1], y_pred[S == 1])
    return abs(tpr0 - tpr1), abs(fpr0 - fpr1), (tpr0, tpr1, fpr0, fpr1)


def equal_opportunity_gap(y_true: np.ndarray, y_pred: np.ndarray, S: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Equal opportunity gap (difference in TPR between groups)."""
    tpr0, _, _ = _rates(y_true[S == 0], y_pred[S == 0])
    tpr1, _, _ = _rates(y_true[S == 1], y_pred[S == 1])
    return abs(tpr0 - tpr1), (tpr0, tpr1)


def fnr_diff(y_true: np.ndarray, y_pred: np.ndarray, S: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Difference in false negative rates between groups."""
    _, _, fnr0 = _rates(y_true[S == 0], y_pred[S == 0])
    _, _, fnr1 = _rates(y_true[S == 1], y_pred[S == 1])
    return abs(fnr0 - fnr1), (fnr0, fnr1)


def nonconformity_binary(proba_pos: np.ndarray, y_true: np.ndarray, *, variant: str = "plain", S: np.ndarray = None, w: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Compute nonconformity scores for binary classification.

    Parameters
    ----------
    proba_pos : array-like
        Predicted probabilities for the positive class.
    y_true : array-like
        True binary labels.
    variant : str, optional
        "plain" uses 1 - p_y; "normalized" divides by sqrt(p*(1-p)).
    S : array-like, optional
        Sensitive attribute for optional reweighting.
    w : tuple
        Group-specific weights when S is provided.
    """
    p1 = np.clip(proba_pos, 1e-6, 1 - 1e-6)
    p0 = 1.0 - p1
    p_y = np.where(y_true == 1, p1, p0)
    if variant == "plain":
        alpha = 1.0 - p_y
    elif variant == "normalized":
        denom = np.sqrt(p1 * (1 - p1))
        denom = np.where(denom < 1e-6, 1e-6, denom)
        alpha = (1.0 - p_y) / denom
    else:
        raise ValueError("Unknown variant")
    if S is not None:
        wg = np.where(S == 0, w[0], w[1])
        alpha = alpha * wg
    return alpha


def fit_marginal_cp(proba_cal: np.ndarray, y_cal: np.ndarray, *, alpha_level: float = 0.1, variant: str = "plain") -> Dict[str, np.ndarray]:
    """Fit marginal (split) conformal predictor for binary classification."""
    alpha_cal = nonconformity_binary(proba_cal, y_cal, variant=variant)
    cal_sorted = np.sort(alpha_cal)
    return {"alpha_level": alpha_level, "cal_sorted": cal_sorted, "variant": variant}


def predict_marginal_cp(cp_obj: Dict[str, np.ndarray], proba_test: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """Predict conformal sets for test probabilities using a marginal CP object."""
    cal_sorted = cp_obj["cal_sorted"]
    alpha_level = cp_obj["alpha_level"]
    variant = cp_obj["variant"]
    n_cal = len(cal_sorted)
    pred_sets = []
    pvals = []
    for p in proba_test:
        p1 = float(np.clip(p, 1e-6, 1 - 1e-6))
        p0 = 1.0 - p1
        if variant == "plain":
            alpha_y1 = 1.0 - p1
            alpha_y0 = 1.0 - p0
        elif variant == "normalized":
            den = max(np.sqrt(p1 * (1 - p1)), 1e-6)
            alpha_y1 = (1.0 - p1) / den
            alpha_y0 = (1.0 - p0) / den
        else:
            raise ValueError("Unknown variant")
        idx_ge1 = np.searchsorted(cal_sorted, alpha_y1, side="left")
        idx_ge0 = np.searchsorted(cal_sorted, alpha_y0, side="left")
        n_ge1 = n_cal - idx_ge1
        n_ge0 = n_cal - idx_ge0
        pval1 = (1 + n_ge1) / (n_cal + 1)
        pval0 = (1 + n_ge0) / (n_cal + 1)
        pred_set = []
        if pval0 > alpha_level:
            pred_set.append(0)
        if pval1 > alpha_level:
            pred_set.append(1)
        pred_sets.append(pred_set)
        pvals.append([pval0, pval1])
    return pred_sets, np.array(pvals)


def fit_mondrian_cp(proba_cal: np.ndarray, y_cal: np.ndarray, S_cal: np.ndarray, *, alpha_level: float = 0.1, variant: str = "plain", group_weights: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, Dict]:
    """Fit group-conditional (Mondrian) conformal predictor for binary classification."""
    alpha_cal = nonconformity_binary(proba_cal, y_cal, variant=variant, S=S_cal, w=group_weights)
    cal_sorted = {}
    for s_val in [0, 1]:
        cal_sorted[s_val] = np.sort(alpha_cal[S_cal == s_val])
    return {
        "alpha_level": alpha_level,
        "cal_sorted_by_s": cal_sorted,
        "variant": variant,
        "w": group_weights,
    }


def predict_mondrian_cp(cp_obj: Dict[str, Dict], proba_test: np.ndarray, S_test: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """Predict conformal sets for test probabilities using a Mondrian CP object."""
    alpha_level = cp_obj["alpha_level"]
    cal_sorted_by_s = cp_obj["cal_sorted_by_s"]
    variant = cp_obj["variant"]
    pred_sets = []
    pvals = []
    for p, s in zip(proba_test, S_test):
        cal_sorted = cal_sorted_by_s[int(s)]
        p1 = float(np.clip(p, 1e-6, 1 - 1e-6))
        p0 = 1.0 - p1
        if variant == "plain":
            alpha_y1 = 1.0 - p1
            alpha_y0 = 1.0 - p0
        elif variant == "normalized":
            den = max(np.sqrt(p1 * (1 - p1)), 1e-6)
            alpha_y1 = (1.0 - p1) / den
            alpha_y0 = (1.0 - p0) / den
        else:
            raise ValueError("Unknown variant")
        n_cal = len(cal_sorted)
        idx_ge1 = np.searchsorted(cal_sorted, alpha_y1, side="left")
        idx_ge0 = np.searchsorted(cal_sorted, alpha_y0, side="left")
        n_ge1 = n_cal - idx_ge1
        n_ge0 = n_cal - idx_ge0
        pval1 = (1 + n_ge1) / (n_cal + 1)
        pval0 = (1 + n_ge0) / (n_cal + 1)
        pred_set = []
        if pval0 > alpha_level:
            pred_set.append(0)
        if pval1 > alpha_level:
            pred_set.append(1)
        pred_sets.append(pred_set)
        pvals.append([pval0, pval1])
    return pred_sets, np.array(pvals)


def evaluate_cp_sets(y_true: np.ndarray, S: np.ndarray, pred_sets: List[List[int]], name: str = "CP") -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate conformal prediction sets and compute fairness metrics on accepted predictions.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    S : array-like
        Sensitive attribute values.
    pred_sets : list of lists
        List of prediction sets per instance.
    name : str
        Label for summary.

    Returns
    -------
    summary : dict
        Summary of coverage, acceptance, fairness and accuracy metrics.
    accept_mask : np.ndarray
        Boolean mask indicating which predictions are singletons (accepted).
    point_pred : np.ndarray
        Point predictions for accepted cases (for rejected cases value is -1).
    """
    set_sizes = np.array([len(s) for s in pred_sets])
    covered = np.array([y_true[i] in pred_sets[i] for i in range(len(y_true))], dtype=int)
    accept_mask = (set_sizes == 1)
    reject_mask = ~accept_mask
    point_pred = np.array([s[0] if len(s) == 1 else -1 for s in pred_sets])
    point_pred_accepted = point_pred[accept_mask]
    y_true_accepted = y_true[accept_mask]
    S_accepted = S[accept_mask]
    cov_overall = float(covered.mean())
    cov_by_group = {
        s_val: float(covered[S == s_val].mean()) if np.any(S == s_val) else np.nan
        for s_val in [0, 1]
    }
    acc_overall = float(accept_mask.mean())
    rej_overall = float(reject_mask.mean())
    acc_by_group = {
        s_val: float(accept_mask[S == s_val].mean()) if np.any(S == s_val) else np.nan
        for s_val in [0, 1]
    }
    rej_by_group = {
        s_val: float(reject_mask[S == s_val].mean()) if np.any(S == s_val) else np.nan
        for s_val in [0, 1]
    }
    if len(point_pred_accepted) > 0:
        dp_gap, p0, p1 = demographic_parity_diff(point_pred_accepted, S_accepted)
        eo_tpr_gap, eo_fpr_gap, eo_detail = equalized_odds_gaps(y_true_accepted, point_pred_accepted, S_accepted)
        eopp_gap, eopp_detail = equal_opportunity_gap(y_true_accepted, point_pred_accepted, S_accepted)
        fnr_gap, fnr_detail = fnr_diff(y_true_accepted, point_pred_accepted, S_accepted)
        from sklearn.metrics import accuracy_score
        acc_on_accepted = float(accuracy_score(y_true_accepted, point_pred_accepted))
    else:
        dp_gap = p0 = p1 = float('nan')
        eo_tpr_gap = eo_fpr_gap = float('nan')
        eopp_gap = float('nan')
        fnr_gap = float('nan')
        acc_on_accepted = float('nan')
    summary = {
        "name": name,
        "coverage_overall": cov_overall,
        "coverage_by_group": cov_by_group,
        "accept_rate_overall": acc_overall,
        "accept_rate_by_group": acc_by_group,
        "reject_rate_overall": rej_overall,
        "reject_rate_by_group": rej_by_group,
        "accepted_count": int(accept_mask.sum()),
        "dp_gap_on_accepted": dp_gap,
        "eo_tpr_gap_on_accepted": eo_tpr_gap,
        "eo_fpr_gap_on_accepted": eo_fpr_gap,
        "eopp_gap_on_accepted": eopp_gap,
        "fnr_gap_on_accepted": fnr_gap,
        "acc_on_accepted": acc_on_accepted,
    }
    return summary, accept_mask, point_pred
