"""Abliacija ir atsparumo bandymai."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEMOGRAPHIC_FEATURES, PAY_STATUS_FEATURES, RANDOM_SEED
from src.metrics import evaluate_probabilities
from src.models import train_xgboost
from src.preprocess import SplitData, select_features


def run_ablation(
    split: SplitData,
    threshold: float,
    c_fn: float,
    c_fp: float,
) -> pd.DataFrame:
    """Požymių grupių pašalinimas: demografija ir mokėjimų būsenos."""
    setups = {
        "visi požymiai": [],
        "be demografijos": DEMOGRAPHIC_FEATURES,
        "be mokėjimų būsenų": PAY_STATUS_FEATURES,
    }
    rows = []
    for name, drop in setups.items():
        X_tr = select_features(split.X_train, drop)
        X_va = select_features(split.X_val, drop)
        X_te = select_features(split.X_test, drop)
        model = train_xgboost(X_tr, split.y_train, X_va, split.y_val)
        p_test = model.predict_proba(X_te)
        metrics = evaluate_probabilities(split.y_test, p_test, threshold, c_fn, c_fp)
        rows.append(
            {
                "variant": name,
                "dropped": ", ".join(drop) if drop else "—",
                "val_pr_auc": model.val_pr_auc,
                "test_pr_auc": metrics["pr_auc"],
                "test_brier": metrics["brier"],
                "test_cost": metrics["cost"],
                "params": model.best_params,
            }
        )
    return pd.DataFrame(rows)


def _add_gaussian_noise(X: pd.DataFrame, scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = X.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    cat = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in numeric]
    numeric = [c for c in numeric if c not in cat]
    for col in numeric:
        std = float(out[col].std() or 0.0)
        if std == 0:
            continue
        out[col] = out[col] + rng.normal(0.0, scale * std, size=len(out))
    return out


def _mask_missing(X: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = X.copy()
    n_cells = out.size
    n_mask = int(n_cells * fraction)
    rows = rng.integers(0, out.shape[0], size=n_mask)
    cols = rng.integers(0, out.shape[1], size=n_mask)
    for r, c in zip(rows, cols):
        out.iat[r, c] = np.nan
    return out


def run_robustness(
    model,
    calibrator,
    split: SplitData,
    threshold: float,
    c_fn: float,
    c_fp: float,
) -> pd.DataFrame:
    """Triukšmas, trūkstamos reikšmės ir grupės poslinkis (vyresni klientai)."""

    def probs(X: pd.DataFrame) -> np.ndarray:
        return calibrator.transform(model.predict_proba(X))

    scenarios = []
    p_clean = probs(split.X_test)
    scenarios.append(("švarus testas", p_clean, split.y_test.to_numpy(), split.X_test))

    noisy = _add_gaussian_noise(split.X_test, scale=0.15, seed=RANDOM_SEED + 1)
    scenarios.append(("Gauso triukšmas σ=0.15·std", probs(noisy), split.y_test.to_numpy(), noisy))

    missing = _mask_missing(split.X_test, fraction=0.10, seed=RANDOM_SEED + 2)
    scenarios.append(("10 % trūkstamų reikšmių", probs(missing), split.y_test.to_numpy(), missing))

    older_mask = split.X_test["AGE"] >= 50
    if older_mask.sum() >= 50:
        X_old = split.X_test.loc[older_mask]
        y_old = split.y_test.loc[older_mask]
        scenarios.append(("grupės poslinkis: amžius ≥ 50", probs(X_old), y_old.to_numpy(), X_old))

    rows = []
    for name, p, y, _X in scenarios:
        m = evaluate_probabilities(y, p, threshold, c_fn, c_fp)
        rows.append(
            {
                "scenario": name,
                "n": m["n"],
                "pr_auc": m["pr_auc"],
                "brier": m["brier"],
                "ece": m["ece"],
                "cost": m["cost"],
                "recall_at_t": m["recall_at_t"],
            }
        )
    return pd.DataFrame(rows)
