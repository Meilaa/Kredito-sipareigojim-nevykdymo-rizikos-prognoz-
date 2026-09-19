"""Duomenų skaidymas ir preprocessing tik pagal mokymo imtį."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_val: pd.Series
    ids_test: pd.Series


def stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> SplitData:
    """Stratifikuotas 70/15/15 skaidymas pagal tikslinę klasę.

    Pirma atskiriamas testas (15 %), tada likusi imtis dalijama į mokymą ir
    validavimą, kad validavimas sudarytų 15 % visos imties.
    """
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
        raise ValueError("Skaidymo santykių suma turi būti 1.")

    ids = df["ID"] if "ID" in df.columns else pd.Series(range(1, len(df) + 1), index=df.index)
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, "ID"}]
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    X_tv, X_test, y_tv, y_test, id_tv, id_test = train_test_split(
        X, y, ids, test_size=TEST_RATIO, stratify=y, random_state=seed
    )
    val_from_tv = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_tv, y_tv, id_tv, test_size=val_from_tv, stratify=y_tv, random_state=seed
    )
    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        ids_train=id_train.reset_index(drop=True),
        ids_val=id_val.reset_index(drop=True),
        ids_test=id_test.reset_index(drop=True),
    )


def _present(columns: list[str], available: pd.Index) -> list[str]:
    return [c for c in columns if c in available]


def build_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool,
    feature_subset: list[str] | None = None,
) -> ColumnTransformer:
    """Preprocessing parametrai skaičiuojami tik iš mokymo imties.

    Medžių modeliams scale_numeric=False, logistinei regresijai True.
    """
    cols = list(X_train.columns if feature_subset is None else feature_subset)
    frame = X_train[cols]
    cat = _present(CATEGORICAL_FEATURES, frame.columns)
    num = _present(NUMERIC_FEATURES, frame.columns)
    num = [c for c in num if c not in cat]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num),
            ("cat", categorical_pipe, cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def select_features(X: pd.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
    drop = drop or []
    keep = [c for c in X.columns if c not in drop]
    return X[keep]
