"""Tikimybių kalibravimas validavimo imtyje (testas nenaudojamas)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.formulas import logistic_probability


class ProbabilityCalibrator:
    """Isotonic arba Platt (sigmoid) kalibravimas, mokomas tik validavimo imtyje."""

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method turi būti 'isotonic' arba 'platt'")
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        p_val = np.asarray(p_val, dtype=float).reshape(-1)
        y_val = np.asarray(y_val).astype(int)
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            self._iso.fit(p_val, y_val)
        else:
            self._platt = LogisticRegression(solver="lbfgs", max_iter=1000)
            self._platt.fit(p_val.reshape(-1, 1), y_val)
        return self

    def transform(self, p_hat: np.ndarray) -> np.ndarray:
        p_hat = np.asarray(p_hat, dtype=float).reshape(-1)
        if self.method == "isotonic":
            if self._iso is None:
                raise RuntimeError("Kalibratorius neapmokytas.")
            return np.asarray(self._iso.transform(p_hat), dtype=float)
        if self._platt is None:
            raise RuntimeError("Kalibratorius neapmokytas.")
        z = self._platt.decision_function(p_hat.reshape(-1, 1))
        return logistic_probability(z)


def choose_calibrator(
    p_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[ProbabilityCalibrator, str, float, float]:
    """Renka kalibratorių pagal mažesnį Brier validavimo imtyje."""
    from src.formulas import brier_score

    candidates = {}
    for method in ("isotonic", "platt"):
        cal = ProbabilityCalibrator(method).fit(p_val, y_val)
        p_cal = cal.transform(p_val)
        candidates[method] = (cal, brier_score(p_cal, y_val))
    raw_brier = brier_score(p_val, y_val)
    best_method = min(candidates, key=lambda m: candidates[m][1])
    cal, cal_brier = candidates[best_method]
    if cal_brier >= raw_brier:
        return IdentityCalibrator(), "none", raw_brier, raw_brier
    return cal, best_method, raw_brier, cal_brier


class IdentityCalibrator(ProbabilityCalibrator):
    def __init__(self) -> None:
        self.method = "none"

    def fit(self, p_val, y_val):
        return self

    def transform(self, p_hat: np.ndarray) -> np.ndarray:
        return np.asarray(p_hat, dtype=float).reshape(-1)
