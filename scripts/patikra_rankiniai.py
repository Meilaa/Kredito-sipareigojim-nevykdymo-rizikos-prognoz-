#!/usr/bin/env python3
"""Patikra: ar rankiniu būdu parašyti algoritmai sutampa su bibliotekomis.

Eksperimente naudojami TIK rankiniai modeliai (`src/rankiniai_modeliai.py`).
Šis skriptas yra atskira patikra, kuri tuos pačius duomenis paduoda ir
bibliotekoms, ir palygina rezultatus. Jei skaičiai sutampa, tai rodo, kad
rankinis įgyvendinimas nėra klaidingas.

Paleidimas:
    python scripts/patikra_rankiniai.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.config import ARTIFACTS_DIR, RANDOM_SEED, TABLES_DIR
from src.data import load_dataset
from src.preprocess import build_preprocessor, stratified_split
from src.rankiniai_modeliai import (
    AtsitiktinisMiskas,
    GradientinisBoostingas,
    LogistineRegresija,
)


def _pr(y, p) -> float:
    return float(average_precision_score(y, p))


def patikrinti() -> dict:
    df, _ = load_dataset()
    split = stratified_split(df, seed=RANDOM_SEED)
    y_tr = split.y_train.to_numpy()
    y_va = split.y_val.to_numpy()

    # Tie patys paruošti duomenys abiem pusėms, kad lygintume tik algoritmus
    prep_su_masteliu = build_preprocessor(split.X_train, scale_numeric=True)
    X_tr_s = prep_su_masteliu.fit_transform(split.X_train)
    X_va_s = prep_su_masteliu.transform(split.X_val)

    prep_be_mastelio = build_preprocessor(split.X_train, scale_numeric=False)
    X_tr = prep_be_mastelio.fit_transform(split.X_train)
    X_va = prep_be_mastelio.transform(split.X_val)

    eilutes = []

    # ------------------------------------------------------------
    # 1. Logistinė regresija: Newton–Raphson vs sklearn lbfgs
    # ------------------------------------------------------------
    t0 = time.time()
    man_log = LogistineRegresija(C=0.1).fit(X_tr_s, y_tr)
    t_man = time.time() - t0
    t0 = time.time()
    sk_log = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(X_tr_s, y_tr)
    t_bib = time.time() - t0
    p_man = man_log.predict_proba(X_va_s)[:, 1]
    p_bib = sk_log.predict_proba(X_va_s)[:, 1]
    eilutes.append(
        {
            "modelis": "Logistinė regresija",
            "rankinis": "Newton–Raphson (src/rankiniai_modeliai.py)",
            "biblioteka": "sklearn LogisticRegression (lbfgs)",
            "pr_auc_rankinis": _pr(y_va, p_man),
            "pr_auc_biblioteka": _pr(y_va, p_bib),
            "maks_tikimybiu_skirtumas": float(np.max(np.abs(p_man - p_bib))),
            "koeficientu_skirtumas": float(np.max(np.abs(man_log.coef_ - sk_log.coef_[0]))),
            "sekundes_rankinis": t_man,
            "sekundes_biblioteka": t_bib,
        }
    )

    # ------------------------------------------------------------
    # 2. Gradientinis boostingas vs XGBoost
    # ------------------------------------------------------------
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    bendri = dict(n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=5)

    t0 = time.time()
    man_gb = GradientinisBoostingas(
        **bendri,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        n_binu=64,
        random_state=RANDOM_SEED,
    ).fit(X_tr, y_tr)
    t_man = time.time() - t0

    t0 = time.time()
    bib_gb = XGBClassifier(
        **bendri,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        scale_pos_weight=spw,
        tree_method="hist",
    ).fit(X_tr, y_tr)
    t_bib = time.time() - t0

    p_man = man_gb.predict_proba(X_va)[:, 1]
    p_bib = bib_gb.predict_proba(X_va)[:, 1]
    eilutes.append(
        {
            "modelis": "XGBoost",
            "rankinis": "g, h ir w = -ΣG/(ΣH+λ) (src/rankiniai_modeliai.py)",
            "biblioteka": "xgboost XGBClassifier (hist)",
            "pr_auc_rankinis": _pr(y_va, p_man),
            "pr_auc_biblioteka": _pr(y_va, p_bib),
            "tikimybiu_koreliacija": float(np.corrcoef(p_man, p_bib)[0, 1]),
            "sekundes_rankinis": t_man,
            "sekundes_biblioteka": t_bib,
        }
    )

    # ------------------------------------------------------------
    # 3. Atsitiktinis miškas vs sklearn RandomForest
    # ------------------------------------------------------------
    t0 = time.time()
    man_rf = AtsitiktinisMiskas(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ).fit(X_tr, y_tr)
    t_man = time.time() - t0

    t0 = time.time()
    bib_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    ).fit(X_tr, y_tr)
    t_bib = time.time() - t0

    p_man = man_rf.predict_proba(X_va)[:, 1]
    p_bib = bib_rf.predict_proba(X_va)[:, 1]
    eilutes.append(
        {
            "modelis": "Random Forest",
            "rankinis": "bootstrap + Gini (src/rankiniai_modeliai.py)",
            "biblioteka": "sklearn RandomForestClassifier",
            "pr_auc_rankinis": _pr(y_va, p_man),
            "pr_auc_biblioteka": _pr(y_va, p_bib),
            "tikimybiu_koreliacija": float(np.corrcoef(p_man, p_bib)[0, 1]),
            "sekundes_rankinis": t_man,
            "sekundes_biblioteka": t_bib,
        }
    )

    lentele = pd.DataFrame(eilutes)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lentele.to_csv(TABLES_DIR / "rankiniu_patikra.csv", index=False)

    payload = {
        "paaiskinimas": (
            "Eksperimente naudojami tik rankiniai modeliai. Ši lentelė yra patikra: "
            "tie patys paruošti duomenys paduoti ir bibliotekoms, palygintos "
            "validavimo imties metrikos."
        ),
        "eilutes": eilutes,
    }
    (ARTIFACTS_DIR / "rankiniu_patikra.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    rez = patikrinti()
    for e in rez["eilutes"]:
        print(f"\n{e['modelis']}")
        print(f"  rankinis    PR-AUC {e['pr_auc_rankinis']:.6f}  ({e['sekundes_rankinis']:.1f} s)")
        print(f"  biblioteka  PR-AUC {e['pr_auc_biblioteka']:.6f}  ({e['sekundes_biblioteka']:.1f} s)")
        if "maks_tikimybiu_skirtumas" in e:
            print(f"  maks. tikimybių skirtumas: {e['maks_tikimybiu_skirtumas']:.2e}")
            print(f"  maks. koeficientų skirtumas: {e['koeficientu_skirtumas']:.2e}")
        if "tikimybiu_koreliacija" in e:
            print(f"  tikimybių koreliacija: {e['tikimybiu_koreliacija']:.4f}")
    print(f"\nĮrašyta: {Path('artifacts/tables/rankiniu_patikra.csv')}")
