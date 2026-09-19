"""Grafikai gynimui: PR, kalibracija, klaidų kaina, grupės."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 13,
    }
)
sns.set_palette("colorblind")


def _save(fig: plt.Figure, name: str) -> str:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, (recall, precision) in curves.items():
        ax.plot(recall, precision, label=name, linewidth=2)
    ax.set_xlabel("Recall (jautrumas default klasei)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return _save(fig, filename)


def plot_calibration(tables: dict[str, pd.DataFrame], title: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="ideali kalibracija")
    for name, table in tables.items():
        ax.plot(table["mean_predicted"], table["mean_observed"], marker="o", label=name)
    ax.set_xlabel("Prognozuota default tikimybė")
    ax.set_ylabel("Stebėtas default dažnis")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return _save(fig, filename)


def plot_cost_curve(curve: pd.DataFrame, best_t: float, title: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(curve["threshold"], curve["cost"], color="#1f4e79", linewidth=2)
    ax.axvline(best_t, color="#b00020", linestyle="--", label=f"pasirinktas t = {best_t:.2f}")
    ax.set_xlabel("Sprendimo slenkstis t")
    ax.set_ylabel("Cost(t) = C_FN·FN + C_FP·FP")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _save(fig, filename)


def plot_group_bars(groups: pd.DataFrame, filename: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    sex = groups[groups["group"] == "lytis"]
    age = groups[groups["group"] == "amžiaus grupė"]
    if not sex.empty:
        axes[0].bar(sex["value"], sex["pr_auc"], color="#1f4e79")
        axes[0].set_title("PR-AUC pagal lytį")
        axes[0].set_ylabel("PR-AUC")
    if not age.empty:
        axes[1].bar(age["value"].astype(str), age["pr_auc"], color="#4c6b8a")
        axes[1].set_title("PR-AUC pagal amžiaus grupę")
        axes[1].set_ylabel("PR-AUC")
    fig.suptitle("Modelio veikimas grupėse (testo imtis)")
    return _save(fig, filename)


def plot_metrics_table_as_bars(summary: pd.DataFrame, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(summary))
    width = 0.38
    ax.bar(x - width / 2, summary["pr_auc"], width, label="PR-AUC", color="#1f4e79")
    ax.bar(x + width / 2, summary["brier"], width, label="Brier (mažiau geriau)", color="#b00020")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=15, ha="right")
    ax.set_title("Modelių palyginimas testo imtyje")
    ax.legend(frameon=False)
    return _save(fig, filename)
