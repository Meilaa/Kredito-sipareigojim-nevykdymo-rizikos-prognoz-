"""UCI duomenų gavimas, kokybės patikra ir stulpelių normalizavimas."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    TARGET_COL,
    UCI_DATASET_ID,
    UCI_PAGE,
)

UCI_X_MAP = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_1",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}


def download_uci_dataset(raw_dir: Path | None = None) -> pd.DataFrame:
    """Atsisiunčia UCI Default of Credit Card Clients rinkinį (id=350).

    Duomenų failo kopijos į egzaminą teikti nereikia – pakanka šios funkcijos.
    Šaltinis: {UCI_PAGE}
    """
    raw_dir = raw_dir or RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "uci_default_credit_card_clients.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()
    frame = features.join(target)
    if "ID" not in frame.columns:
        frame.insert(0, "ID", range(1, len(frame) + 1))
    frame.to_csv(cache_path, index=False)
    return frame


def _normalize_columns(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip()
        lower = key.lower().replace(" ", "_").replace(".", "_")
        if lower in {"default_payment_next_month", "y", "dpnm"}:
            rename[col] = TARGET_COL
        elif key == "PAY_0":
            rename[col] = "PAY_1"
        elif key == "PAY_1" and "PAY_0" not in df.columns:
            rename[col] = "PAY_1"
    out = df.rename(columns=rename).copy()
    out = out.rename(columns={k: v for k, v in UCI_X_MAP.items() if k in out.columns})
    if require_target and TARGET_COL not in out.columns:
        raise ValueError(
            f"Nerastas tikslinis stulpelis. Turimi: {list(out.columns)}"
        )
    return out


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Sutraukia retas / nedokumentuotas EDUCATION ir MARRIAGE reikšmes.

    EDUCATION 0, 5, 6 -> 4 (kita / nežinoma)
    MARRIAGE 0 -> 3 (kita / nežinoma)
    """
    out = df.copy()
    if "EDUCATION" in out.columns:
        out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in out.columns:
        out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})
    return out


def quality_report(df: pd.DataFrame) -> dict:
    y = df[TARGET_COL].astype(int)
    n = len(df)
    n_default = int(y.sum())
    missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
    duplicates = int(df.duplicated().sum())
    return {
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        "n_default": n_default,
        "n_non_default": n - n_default,
        "default_rate": float(y.mean()),
        "missing_by_column": missing,
        "duplicate_rows": duplicates,
        "source": UCI_PAGE,
        "uci_id": UCI_DATASET_ID,
    }


def load_dataset(raw_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = download_uci_dataset(raw_dir)
    df = _normalize_columns(raw)
    df = clean_categories(df)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    report = quality_report(df)
    return df, report
