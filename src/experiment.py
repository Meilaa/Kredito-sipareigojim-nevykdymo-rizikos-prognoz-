"""Pagrindinis atkuriamas eksperimentas.

Viena komanda: python run_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ablation import run_ablation, run_robustness
from src.calibrate import choose_calibrator
from src.config import (
    ARTIFACTS_DIR,
    C_FN,
    C_FP,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.data import load_dataset
from src.evaluate import error_examples, error_profile, group_performance
from src.metrics import (
    calibration_table,
    evaluate_probabilities,
    select_threshold_by_cost,
)
from src.models import (
    logistic_formula_check,
    train_constant,
    train_logistic,
    train_random_forest,
    train_xgboost,
)
from src.plots import (
    plot_calibration,
    plot_cost_curve,
    plot_group_bars,
    plot_metrics_table_as_bars,
    plot_pr_curves,
)
from src.preprocess import stratified_split


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Objektas neserializuojamas: {type(obj)}")


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _metrics_for_table(metrics: dict) -> dict:
    skip = {"pr_curve_precision", "pr_curve_recall"}
    return {k: v for k, v in metrics.items() if k not in skip}


def run_experiment(seed: int = RANDOM_SEED, c_fn: float = C_FN, c_fp: float = C_FP) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df, quality = load_dataset()
    split = stratified_split(df, seed=seed)
    split_info = {
        "seed": seed,
        "n_train": int(len(split.X_train)),
        "n_val": int(len(split.X_val)),
        "n_test": int(len(split.X_test)),
        "train_default_rate": float(split.y_train.mean()),
        "val_default_rate": float(split.y_val.mean()),
        "test_default_rate": float(split.y_test.mean()),
        "note": "Preprocessing parametrai skaičiuojami tik mokymo imtyje. Hiperparametrai ir slenkstis – tik validavime.",
    }

    models = {
        "Bendras dažnis": train_constant(split.y_train),
        "Logistinė regresija": train_logistic(split.X_train, split.y_train, split.X_val, split.y_val),
        "Random Forest": train_random_forest(split.X_train, split.y_train, split.X_val, split.y_val),
        "XGBoost": train_xgboost(split.X_train, split.y_train, split.X_val, split.y_val),
    }

    formula_check = logistic_formula_check(models["Logistinė regresija"], split.X_val, n=8)

    val_scores = {}
    test_scores_raw = {}
    for name, model in models.items():
        val_scores[name] = model.predict_proba(split.X_val)
        test_scores_raw[name] = model.predict_proba(split.X_test)

    calibrated = {}
    calibration_choice = {}
    for name, p_val in val_scores.items():
        cal, method, raw_brier, cal_brier = choose_calibrator(p_val, split.y_val.to_numpy())
        calibrated[name] = cal
        calibration_choice[name] = {
            "method": method,
            "val_brier_raw": raw_brier,
            "val_brier_cal": cal_brier,
        }

    test_scores_cal = {
        name: calibrated[name].transform(p) for name, p in test_scores_raw.items()
    }
    val_scores_cal = {
        name: calibrated[name].transform(p) for name, p in val_scores.items()
    }

    thresholds = {}
    cost_curves = {}
    for name, p_val in val_scores_cal.items():
        t, curve = select_threshold_by_cost(split.y_val.to_numpy(), p_val, c_fn, c_fp)
        thresholds[name] = t
        cost_curves[name] = curve
        curve.to_csv(TABLES_DIR / f"cost_curve_{name.replace(' ', '_')}.csv", index=False)

    rows = []
    test_eval = {}
    for name in models:
        t = thresholds[name]
        raw = evaluate_probabilities(split.y_test, test_scores_raw[name], t, c_fn, c_fp)
        cal = evaluate_probabilities(split.y_test, test_scores_cal[name], t, c_fn, c_fp)
        test_eval[name] = {"raw": raw, "calibrated": cal}
        rows.append(
            {
                "model": name,
                "kind": models[name].kind,
                "params": json.dumps(models[name].best_params, ensure_ascii=False),
                "val_pr_auc_for_selection": models[name].val_pr_auc if models[name].kind != "constant" else float(split.y_val.mean()),
                "calibration": calibration_choice[name]["method"],
                "threshold": t,
                "pr_auc": cal["pr_auc"],
                "roc_auc": cal["roc_auc"],
                "brier": cal["brier"],
                "ece": cal["ece"],
                "accuracy": cal["accuracy"],
                "cost": cal["cost"],
                "recall_at_t": cal["recall_at_t"],
                "precision_at_t": cal["precision_at_t"],
                "FP": cal["FP"],
                "FN": cal["FN"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    winner_name = str(summary.iloc[0]["model"])
    winner = models[winner_name]
    winner_t = thresholds[winner_name]
    winner_p = test_scores_cal[winner_name]

    groups = group_performance(split.X_test, split.y_test.to_numpy(), winner_p, winner_t)
    errors = error_examples(split.X_test, split.y_test.to_numpy(), winner_p, split.ids_test, winner_t)
    profile = error_profile(split.X_test, split.y_test.to_numpy(), winner_p, winner_t)
    ablation = run_ablation(split, winner_t, c_fn, c_fp)
    robustness = run_robustness(winner, calibrated[winner_name], split, winner_t, c_fn, c_fp)

    summary.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    groups.to_csv(TABLES_DIR / "group_performance.csv", index=False)
    profile.to_csv(TABLES_DIR / "error_profile.csv", index=False)
    ablation.to_csv(TABLES_DIR / "ablation.csv", index=False)
    robustness.to_csv(TABLES_DIR / "robustness.csv", index=False)
    errors["false_negative"].to_csv(TABLES_DIR / "fn_examples.csv", index=False)
    errors["false_positive"].to_csv(TABLES_DIR / "fp_examples.csv", index=False)

    pr_curves = {}
    cal_tables = {}
    for name in models:
        ev = test_eval[name]["calibrated"]
        pr_curves[name] = (np.array(ev["pr_curve_recall"]), np.array(ev["pr_curve_precision"]))
        cal_tables[name] = calibration_table(split.y_test, test_scores_cal[name])
        cal_tables[name].to_csv(TABLES_DIR / f"calibration_{name.replace(' ', '_')}.csv", index=False)

    plot_paths = {
        "pr": plot_pr_curves(pr_curves, "PR kreivės testo imtyje (kalibruotos tikimybės)", "pr_curves.png"),
        "calibration": plot_calibration(cal_tables, "Kalibracijos kreivės testo imtyje", "calibration.png"),
        "cost": plot_cost_curve(
            cost_curves[winner_name],
            winner_t,
            f"Klaidų kainos kreivė: {winner_name} (C_FN={c_fn}, C_FP={c_fp})",
            "cost_curve.png",
        ),
        "groups": plot_group_bars(groups, "group_prauc.png"),
        "bars": plot_metrics_table_as_bars(summary, "model_bars.png"),
    }

    predictions = pd.DataFrame(
        {
            "ID": split.ids_test,
            "y": split.y_test,
            **{f"p_raw_{k}": v for k, v in test_scores_raw.items()},
            **{f"p_cal_{k}": v for k, v in test_scores_cal.items()},
        }
    )
    predictions.to_csv(TABLES_DIR / "test_predictions.csv", index=False)

    val_predictions = pd.DataFrame(
        {
            "ID": split.ids_val,
            "y": split.y_val,
            **{f"p_cal_{k}": v for k, v in val_scores_cal.items()},
        }
    )
    val_predictions.to_csv(TABLES_DIR / "val_predictions.csv", index=False)

    for name, model in models.items():
        safe = name.replace(" ", "_")
        joblib.dump(
            {
                "model": model,
                "calibrator": calibrated[name],
                "threshold": thresholds[name],
            },
            MODELS_DIR / f"{safe}.joblib",
        )
    joblib.dump(split.X_test.head(1), MODELS_DIR / "schema_example.joblib")

    payload = {
        "seed": seed,
        "c_fn": c_fn,
        "c_fp": c_fp,
        "quality": quality,
        "split": split_info,
        "formula_check": formula_check,
        "calibration_choice": calibration_choice,
        "thresholds": thresholds,
        "summary": summary.to_dict(orient="records"),
        "winner": winner_name,
        "winner_notes": winner.notes,
        "winner_params": winner.best_params,
        "logistic_extra": models["Logistinė regresija"].extra,
        "test_eval": {
            name: {"raw": _metrics_for_table(vals["raw"]), "calibrated": _metrics_for_table(vals["calibrated"])}
            for name, vals in test_eval.items()
        },
        "groups": groups.to_dict(orient="records"),
        "ablation": ablation.to_dict(orient="records"),
        "robustness": robustness.to_dict(orient="records"),
        "error_profile": profile.to_dict(orient="records"),
        "fn_examples": errors["false_negative"].to_dict(orient="records"),
        "fp_examples": errors["false_positive"].to_dict(orient="records"),
        "plot_paths": plot_paths,
        "hypothesis": (
            "XGBoost dėl netiesinių ryšių ir požymių sąveikų turėtų pranokti "
            "logistinę regresiją pagal PR-AUC, išlaikant priimtiną kalibraciją po kalibravimo."
        ),
    }
    _dump_json(ARTIFACTS_DIR / "results.json", payload)

    print("Eksperimentas baigtas.")
    print(summary[["model", "pr_auc", "brier", "ece", "cost", "threshold"]].to_string(index=False))
    print(f"Laimėtojas pagal PR-AUC: {winner_name}")
    print(f"Formulės ir sklearn tikimybių maks. skirtumas: {formula_check['max_abs_diff']:.2e}")
    return payload


if __name__ == "__main__":
    run_experiment()
