"""Grupių analizė ir klaidų pavyzdžiai."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import EDUCATION_MAP, MARRIAGE_MAP, SEX_MAP
from src.formulas import brier_score, confusion_counts, decision_from_threshold


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=[0, 29, 39, 49, 120],
        labels=["iki 29", "30–39", "40–49", "50+"],
        right=True,
    )


def group_performance(
    X: pd.DataFrame,
    y_true: np.ndarray,
    p_hat: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat, dtype=float)
    y_pred = decision_from_threshold(p_hat, threshold)
    frame = X.copy()
    frame["y"] = y_true
    frame["p"] = p_hat
    frame["yhat"] = y_pred
    frame["SEX_label"] = frame["SEX"].map(SEX_MAP).fillna("kita")
    frame["EDU_label"] = frame["EDUCATION"].map(EDUCATION_MAP).fillna("kita")
    frame["MAR_label"] = frame["MARRIAGE"].map(MARRIAGE_MAP).fillna("kita")
    frame["AGE_band"] = age_band(frame["AGE"])

    rows = []
    for group_name, col in [
        ("lytis", "SEX_label"),
        ("išsilavinimas", "EDU_label"),
        ("šeiminė padėtis", "MAR_label"),
        ("amžiaus grupė", "AGE_band"),
    ]:
        for value, part in frame.groupby(col, observed=True):
            if len(part) < 30:
                continue
            counts = confusion_counts(part["y"].to_numpy(), part["yhat"].to_numpy())
            try:
                pr = float(average_precision_score(part["y"], part["p"]))
            except ValueError:
                pr = float("nan")
            rows.append(
                {
                    "group": group_name,
                    "value": str(value),
                    "n": int(len(part)),
                    "default_rate": float(part["y"].mean()),
                    "mean_p": float(part["p"].mean()),
                    "pr_auc": pr,
                    "brier": brier_score(part["p"].to_numpy(), part["y"].to_numpy()),
                    "fn_rate": counts["FN"] / max(int(part["y"].sum()), 1),
                    "fp_rate": counts["FP"] / max(int((part["y"] == 0).sum()), 1),
                    **counts,
                }
            )
    return pd.DataFrame(rows)


def error_examples(
    X: pd.DataFrame,
    y_true: np.ndarray,
    p_hat: np.ndarray,
    ids: pd.Series,
    threshold: float,
    n: int = 8,
) -> dict[str, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat, dtype=float)
    y_pred = decision_from_threshold(p_hat, threshold)
    frame = X.copy()
    frame["ID"] = ids.to_numpy()
    frame["y"] = y_true
    frame["p"] = p_hat
    frame["yhat"] = y_pred
    cols = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_1",
        "BILL_AMT1",
        "PAY_AMT1",
        "y",
        "p",
        "yhat",
    ]
    cols = [c for c in cols if c in frame.columns]
    fn = frame[(frame["y"] == 1) & (frame["yhat"] == 0)].sort_values("p")
    fp = frame[(frame["y"] == 0) & (frame["yhat"] == 1)].sort_values("p", ascending=False)
    return {
        "false_negative": fn[cols].head(n).reset_index(drop=True),
        "false_positive": fp[cols].head(n).reset_index(drop=True),
    }


def error_profile(X: pd.DataFrame, y_true: np.ndarray, p_hat: np.ndarray, threshold: float) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat, dtype=float)
    y_pred = decision_from_threshold(p_hat, threshold)
    frame = X.copy()
    frame["error_type"] = np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="kita",
    )
    num_cols = [c for c in ["LIMIT_BAL", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1"] if c in frame.columns]
    return frame.groupby("error_type")[num_cols].mean().reset_index()
