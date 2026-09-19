"""Baseline ir intelektualieji modeliai.

Baseline: bendras default dažnis ir logistinė regresija.
Intelektualūs metodai: XGBoost ir Random Forest.
Hiperparametrai parenkami tik validavimo imtyje.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_SEED
from src.formulas import logistic_probability, linear_score
from src.preprocess import build_preprocessor


@dataclass
class TrainedModel:
    name: str
    kind: str
    pipeline: Any
    best_params: dict
    val_pr_auc: float
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "constant":
            return np.full(len(X), float(self.extra["p"]), dtype=float)
        proba = self.pipeline.predict_proba(X)
        return proba[:, 1]


class ConstantRateBaseline:
    """Naivus baseline: visiems klientams priskiria mokymo imties default dažnį."""

    def __init__(self) -> None:
        self.p = 0.0

    def fit(self, y_train: pd.Series) -> "ConstantRateBaseline":
        self.p = float(np.mean(y_train))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p, dtype=float)


def _pr_auc(y_true, p_hat) -> float:
    return float(average_precision_score(y_true, p_hat))


def _fit_search(
    name: str,
    kind: str,
    builder,
    grid: list[dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_numeric: bool,
) -> TrainedModel:
    best: TrainedModel | None = None
    for params in grid:
        preprocessor = build_preprocessor(X_train, scale_numeric=scale_numeric)
        clf = builder(params)
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:, 1]
        score = _pr_auc(y_val, p_val)
        candidate = TrainedModel(
            name=name,
            kind=kind,
            pipeline=pipe,
            best_params=params,
            val_pr_auc=score,
        )
        if best is None or score > best.val_pr_auc:
            best = candidate
    assert best is not None
    return best


def train_constant(y_train: pd.Series) -> TrainedModel:
    model = ConstantRateBaseline().fit(y_train)
    return TrainedModel(
        name="Bendras dažnis",
        kind="constant",
        pipeline=model,
        best_params={"p": model.p},
        val_pr_auc=float("nan"),
        notes="Naivus baseline be požymių.",
        extra={"p": model.p},
    )


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> TrainedModel:
    grid = [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ]

    def builder(params: dict):
        return LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )

    model = _fit_search(
        "Logistinė regresija",
        "logistic",
        builder,
        grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scale_numeric=True,
    )
    model.notes = (
        "Baseline, kurio tikimybė skaičiuojama formule p̂ = 1/(1+e^{-z}), "
        "z = β₀ + Xβ. Koeficientai yra sklearn LogisticRegression.coef_."
    )
    clf = model.pipeline.named_steps["clf"]
    model.extra = {
        "intercept": float(clf.intercept_[0]),
        "coefficients": clf.coef_[0].tolist(),
        "feature_names": list(model.pipeline.named_steps["prep"].get_feature_names_out()),
    }
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> TrainedModel:
    grid = [
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
    ]

    def builder(params: dict):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    model = _fit_search(
        "Random Forest",
        "random_forest",
        builder,
        grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scale_numeric=False,
    )
    model.notes = "Medžių ansamblis; tikimybė – teigiamos klasės lapų dažnių vidurkis."
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> TrainedModel:
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    spw = n_neg / max(n_pos, 1.0)
    grid = [
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "min_child_weight": 3},
        {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 1},
    ]

    def builder(params: dict):
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            scale_pos_weight=spw,
            tree_method="hist",
        )

    model = _fit_search(
        "XGBoost",
        "xgboost",
        builder,
        grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scale_numeric=False,
    )
    model.notes = (
        "Pagrindinis intelektualus metodas. binary:logistic reiškia "
        "p̂ = σ(Σ_k f_k(x)), kur f_k – k-asis medis."
    )
    model.extra["scale_pos_weight"] = spw
    return model


def logistic_formula_check(model: TrainedModel, X: pd.DataFrame, n: int = 5) -> dict:
    """Patikrina, kad sklearn tikimybės sutampa su rankine formule p̂ = σ(z)."""
    if model.kind != "logistic":
        raise ValueError("Formulės patikra skirta logistinei regresijai.")
    prep = model.pipeline.named_steps["prep"]
    clf = model.pipeline.named_steps["clf"]
    X_trans = prep.transform(X.iloc[:n])
    z = linear_score(X_trans, float(clf.intercept_[0]), clf.coef_[0])
    p_formula = logistic_probability(z)
    p_sklearn = model.predict_proba(X.iloc[:n])
    return {
        "z": z.tolist(),
        "p_formula": p_formula.tolist(),
        "p_sklearn": p_sklearn.tolist(),
        "max_abs_diff": float(np.max(np.abs(p_formula - p_sklearn))),
    }
