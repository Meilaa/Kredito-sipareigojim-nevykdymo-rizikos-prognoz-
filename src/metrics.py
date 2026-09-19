"""Vertinimo metrikos: PR-AUC, Brier, kalibracija, kaina."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.formulas import (
    brier_score,
    confusion_counts,
    decision_from_threshold,
    expected_calibration_error,
    total_error_cost,
)


def evaluate_probabilities(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    threshold: float,
    c_fn: float,
    c_fp: float,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat, dtype=float)
    y_pred = decision_from_threshold(p_hat, threshold)
    counts = confusion_counts(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, p_hat)
    return {
        "pr_auc": float(average_precision_score(y_true, p_hat)),
        "roc_auc": float(roc_auc_score(y_true, p_hat)),
        "brier": brier_score(p_hat, y_true),
        "ece": expected_calibration_error(p_hat, y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "cost": total_error_cost(y_true, y_pred, c_fn, c_fp),
        "precision_at_t": float(counts["TP"] / max(counts["TP"] + counts["FP"], 1)),
        "recall_at_t": float(counts["TP"] / max(counts["TP"] + counts["FN"], 1)),
        **counts,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "pr_curve_precision": precision.tolist(),
        "pr_curve_recall": recall.tolist(),
    }


def cost_curve(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    c_fn: float,
    c_fp: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)
    rows = []
    for t in thresholds:
        y_pred = decision_from_threshold(p_hat, float(t))
        counts = confusion_counts(y_true, y_pred)
        rows.append(
            {
                "threshold": float(t),
                "cost": total_error_cost(y_true, y_pred, c_fn, c_fp),
                **counts,
            }
        )
    return pd.DataFrame(rows)


def select_threshold_by_cost(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    c_fn: float,
    c_fp: float,
) -> tuple[float, pd.DataFrame]:
    curve = cost_curve(y_true, p_hat, c_fn, c_fp)
    best_idx = int(curve["cost"].idxmin())
    return float(curve.loc[best_idx, "threshold"]), curve


def calibration_table(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        right_closed = i == n_bins - 1
        mask = (p_hat >= bins[i]) & (
            p_hat <= bins[i + 1] if right_closed else p_hat < bins[i + 1]
        )
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": i,
                "left": float(bins[i]),
                "right": float(bins[i + 1]),
                "mean_predicted": float(np.mean(p_hat[mask])),
                "mean_observed": float(np.mean(y_true[mask])),
                "n": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)
