"""Pagrindinės egzamino formulės, susietos su konkrečiomis kodo vietomis.

Logistinė regresija:
    z_i = β₀ + β₁x_i1 + ... + β_p x_ip
    p̂_i = 1 / (1 + e^{-z_i})

Sprendimo taisyklė:
    ŷ_i = 1, kai p̂_cal,i ≥ t; kitu atveju ŷ_i = 0

Brier score:
    BS = (1/N) Σ_i (p̂_i − y_i)²

Klaidų kaina:
    Cost(t) = C_FN · FN(t) + C_FP · FP(t)
"""

from __future__ import annotations

import numpy as np


def linear_score(X: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    """z_i = β₀ + X_i · β.

    X – i-ojo kliento požymių vektorius (jau po preprocessing),
    intercept – β₀, coefficients – β vektorius.
    """
    X = np.asarray(X, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    return intercept + X @ coefficients


def logistic_probability(z: np.ndarray) -> np.ndarray:
    """p̂_i = 1 / (1 + e^{-z_i}).

    Skaičiuojama skaitiškai stabiliai, kad didelės |z| reikšmės
    nesukeltų overflow.
    """
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def decision_from_threshold(p_cal: np.ndarray, threshold: float) -> np.ndarray:
    """ŷ_i = 1, kai p̂_cal,i ≥ t; kitu atveju ŷ_i = 0."""
    p_cal = np.asarray(p_cal, dtype=float)
    return (p_cal >= threshold).astype(int)


def brier_score(p_hat: np.ndarray, y_true: np.ndarray) -> float:
    """BS = (1/N) Σ_i (p̂_i − y_i)². Mažesnė reikšmė – tikslesnės tikimybės."""
    p_hat = np.asarray(p_hat, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((p_hat - y_true) ** 2))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def total_error_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_fn: float,
    c_fp: float,
) -> float:
    """Cost(t) = C_FN · FN(t) + C_FP · FP(t)."""
    counts = confusion_counts(y_true, y_pred)
    return float(c_fn * counts["FN"] + c_fp * counts["FP"])


def expected_calibration_error(
    p_hat: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Vidutinis |prognozuota tikimybė − stebėtas dažnis| pagal binus, svertas binų dydžiu."""
    p_hat = np.asarray(p_hat, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (p_hat >= bins[i]) & (p_hat < bins[i + 1] if i < n_bins - 1 else p_hat <= bins[i + 1])
        if not np.any(mask):
            continue
        avg_pred = float(np.mean(p_hat[mask]))
        avg_true = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(avg_pred - avg_true)
    return float(ece)
