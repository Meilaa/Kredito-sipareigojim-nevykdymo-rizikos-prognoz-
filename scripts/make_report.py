#!/usr/bin/env python3
"""Sudaro egzamino PDF ataskaitą: užduotis, dokumentacija, lentelės, grafikai."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_texts import lt, surinkti  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
TABLES = ARTIFACTS / "tables"
REPORTS = ROOT / "reports"
OUT = REPORTS / "Egzamino_ataskaita.pdf"
FONT_DIR = ROOT / ".venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf"


def load_results() -> dict:
    return json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))


def extra_plots() -> dict[str, Path]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    paths = {}
    summary = pd.read_csv(TABLES / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(summary))
    ax.bar([i - 0.2 for i in x], summary["pr_auc"], 0.4, label="PR-AUC", color="#1B2A4A")
    ax.bar([i + 0.2 for i in x], summary["brier"], 0.4, label="Brier", color="#8B1E3F")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["model"], rotation=12, ha="right")
    ax.set_ylabel("Reikšmė")
    ax.set_title("Modelių palyginimas testo imtyje")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["compare"] = PLOTS / "report_compare.png"
    fig.savefig(paths["compare"], dpi=170)
    plt.close(fig)

    ab = pd.read_csv(TABLES / "ablation.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(ab["variant"], ab["test_pr_auc"], color="#1B2A4A")
    ax.set_ylabel("PR-AUC teste")
    ax.set_title("Abliacija: gradientinis boostingas be požymių grupių")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["ablation"] = PLOTS / "report_ablation.png"
    fig.savefig(paths["ablation"], dpi=170)
    plt.close(fig)

    rob = pd.read_csv(TABLES / "robustness.csv")
    laimetojas = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))["winner"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(rob["scenario"], rob["pr_auc"], color="#4C6B8A")
    ax.set_xlabel("PR-AUC")
    ax.set_title(f"Atsparumo bandymai ({laimetojas}, kalibruotos tikimybės)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    paths["robust"] = PLOTS / "report_robust.png"
    fig.savefig(paths["robust"], dpi=170)
    plt.close(fig)
    return paths


class Report(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4", unit="mm")
        self.set_margins(18, 16, 18)
        self.set_auto_page_break(auto=True, margin=18)
        regular = FONT_DIR / "DejaVuSans.ttf"
        bold = FONT_DIR / "DejaVuSans-Bold.ttf"
        mono = FONT_DIR / "DejaVuSansMono.ttf"
        self.add_font("DejaVu", "", str(regular))
        self.add_font("DejaVu", "B", str(bold))
        self.add_font("DejaVuMono", "", str(mono))
        self.set_title("Kredito įsipareigojimų nevykdymo rizikos prognozė")
        self.set_author("Meila Andriuškevičiūtė")

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8)
        self.set_text_color(90, 101, 119)
        self.cell(0, 7, "Intelektualiosios sistemos · galutinis egzaminas · DISfm-26", align="L")
        self.ln(9)
        self.set_draw_color(27, 42, 74)
        self.line(18, 15, 192, 15)
        self.set_text_color(27, 42, 74)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(90, 101, 119)
        self.cell(0, 8, str(self.page_no()), align="C")

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "", 10)
        self.set_text_color(27, 42, 74)
        self.multi_cell(0, 5.4, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.2)

    def h1(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(27, 42, 74)
        self.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text: str) -> None:
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "B", 11.5)
        self.set_text_color(139, 30, 63)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(27, 42, 74)
        self.ln(1)

    def bullet(self, items: list[str]) -> None:
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "", 10)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.3, f"•  {item}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def code(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("DejaVuMono", "", 8)
        self.set_fill_color(244, 241, 236)
        self.multi_cell(0, 4.6, text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def caption(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "", 8.5)
        self.set_text_color(90, 101, 119)
        self.multi_cell(0, 4.6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(27, 42, 74)
        self.ln(2)

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
        usable = 174
        if col_widths is None:
            col_widths = [usable / len(headers)] * len(headers)
        self.set_font("DejaVu", "B", 8)
        self.set_fill_color(27, 42, 74)
        self.set_text_color(255, 255, 255)
        self.set_x(self.l_margin)
        for w, h in zip(col_widths, headers):
            self.cell(w, 7, h, border=0, fill=True)
        self.ln()
        self.set_x(self.l_margin)
        self.set_text_color(27, 42, 74)
        self.set_font("DejaVu", "", 8)
        fill = False
        for row in rows:
            self.set_fill_color(244, 241, 236) if fill else self.set_fill_color(255, 255, 255)
            # wrap-aware height
            line_h = 5.2
            n_lines = 1
            for w, val in zip(col_widths, row):
                n_lines = max(n_lines, max(1, self.get_string_width(val) // max(w - 1, 8) + 1))
            n_lines = int(min(n_lines, 3))
            y0 = self.get_y()
            x0 = self.get_x()
            h = line_h * n_lines + 1.5
            if y0 + h > 277:
                self.add_page()
                y0 = self.get_y()
            for w, val in zip(col_widths, row):
                self.set_xy(x0, y0)
                self.rect(x0, y0, w, h, style="F")
                self.multi_cell(w, line_h, val)
                x0 += w
            self.set_xy(18, y0 + h)
            fill = not fill
        self.ln(3)

    def figure(self, path: Path, title: str, w: float = 165) -> None:
        if not path.exists():
            return
        from PIL import Image as PILImage

        im = PILImage.open(path)
        h_mm = w * im.height / im.width
        title_h = 8
        if self.get_y() + title_h + min(h_mm, 120) > 268:
            self.add_page()
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "B", 10)
        self.multi_cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        max_h = 268 - self.get_y()
        if h_mm > max_h:
            w = w * max_h / h_mm
            h_mm = max_h
        self.image(str(path), w=w)
        self.ln(2)


def build() -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    extra = extra_plots()
    res = load_results()
    summary = pd.read_csv(TABLES / "model_comparison.csv")
    groups = pd.read_csv(TABLES / "group_performance.csv")
    ablation = pd.read_csv(TABLES / "ablation.csv")
    robust = pd.read_csv(TABLES / "robustness.csv")
    profile = pd.read_csv(TABLES / "error_profile.csv")
    fn = pd.read_csv(TABLES / "fn_examples.csv")
    fp = pd.read_csv(TABLES / "fp_examples.csv")
    T = surinkti()
    pdf = Report()

    # Titulinis
    pdf.add_page()
    pdf.set_y(48)
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 8, "VILNIUS TECH", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 7, "Intelektualiosios sistemos", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(18)
    pdf.set_font("DejaVu", "B", 18)
    pdf.multi_cell(0, 9, "Kredito įsipareigojimų nevykdymo rizikos prognozė", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 12)
    pdf.multi_cell(0, 7, "Galutinio egzamino ataskaita", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(18)
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 7, "Studentė: Meila Andriuškevičiūtė", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 7, "Grupė: DISfm-26", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 7, "2026–2027 m. m.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(22)
    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(
        0,
        6,
        "UCI Default of Credit Card Clients · kalibruota default tikimybė · "
        "PR-AUC, Brier score, kalibracija, klaidų kaina",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # 1. Užduotis
    pdf.add_page()
    pdf.h1("1. Užduotis")
    pdf.body(
        "Individuali 2026–2027 m. užduotis dalykui „Intelektualiosios sistemos“. "
        "Tema: kredito įsipareigojimų nevykdymo rizika. Finansų įstaiga nori numatyti "
        "kito mėnesio įsipareigojimų nevykdymo riziką."
    )
    pdf.h2("Duomenys ir sprendimo tikslas")
    pdf.bullet(
        [
            "Duomenų rinkinys: UCI Default of Credit Card Clients, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "Prognozuojamas rezultatas: kalibruota įsipareigojimų nevykdymo tikimybė.",
            "Privalomas baseline: bendras dažnis ir logistinė regresija.",
            T["intelektualus_metodai"],
            "Pagrindinės vertinimo kryptys: PR-AUC, Brier score, kalibracija ir klaidų kaštų kreivė.",
            "Duomenų skaidymas: stratifikuotas testas; preprocessing tik mokymo imtyje.",
            "Svarbiausias probleminis aspektas: teisingumas skirtingoms grupėms ir tikimybės kalibravimas.",
        ]
    )
    pdf.h2("Galutinio egzamino reikalavimai")
    pdf.bullet(
        [
            "Įgyvendinti duomenų paruošimo grandinę, baseline ir bent du intelektualiuosius metodus.",
            "Bent vienam modeliui parodyti formulės ar taisyklės ryšį su konkrečia kodo vieta. "
            "Šiame darbe visi mokymo algoritmai parašyti rankiniu būdu.",
            "Palyginti metodus tuo pačiu skaidymu ir tomis pačiomis metrikomis. Hiperparametrų parinkimui nenaudoti galutinio testo.",
            "Atlikti bent vieną abliaciją ir bent vieną atsparumo bandymą.",
            "Pateikti rezultatų lentelę, grafiką, klaidų pavyzdžius ir paaiškinti, kodėl metodas laimėjo.",
            "Pateikti paleidimo instrukciją, priklausomybes, fiksuotas sėklas ir vieną komandą eksperimentui.",
            "Pateikti AI naudojimo žurnalą.",
        ]
    )

    # 2. Dokumentacija
    pdf.h1("2. Sprendimo dokumentacija")
    pdf.h2("2.1. Problema")
    pdf.body(T["problema"])
    pdf.h2("2.2. Hipotezė")
    pdf.body(res["hypothesis"])
    pdf.body(T["hipotezes_isvada"])
    pdf.h2("2.3. Įvestis ir išvestis")
    pdf.body(
        "Įvestis X: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, mokėjimų būsenos PAY_1–PAY_6, "
        "sąskaitos BILL_AMT1–BILL_AMT6, mokėjimai PAY_AMT1–PAY_AMT6. Tikslas Y: 0 – "
        "įsipareigojimai vykdomi, 1 – kitą mėnesį nevykdomi. Išvestis: p̂ ∈ [0, 1], po "
        "kalibravimo p̂_cal, klasė ŷ = 1, jei p̂_cal ≥ t."
    )
    pdf.h2("2.4. Duomenų grandinė")
    pdf.body(T["duomenu_grandine"])
    pdf.code(T["grandines_kodas"])
    pdf.h2("2.5. Modeliai")
    pdf.body(T["modeliai"])
    pdf.h2("2.6. Formulės ir kodas")
    pdf.body("Logistinė regresija:")
    pdf.code("z_i = β0 + β1 x_i1 + … + βp x_ip\np̂_i = 1 / (1 + e^{-z_i})")
    pdf.body(T["formules_patikra"])
    pdf.body("Sprendimo taisyklė, Brier score ir klaidų kaina:")
    pdf.code(
        "ŷ_i = 1, kai p̂_cal,i ≥ t; kitaip ŷ_i = 0\n"
        "BS = (1/N) Σ (p̂_i − y_i)²\n"
        f"Cost(t) = C_FN · FN(t) + C_FP · FP(t),  C_FN = {int(res['c_fn'])}, C_FP = {int(res['c_fp'])}"
    )
    pdf.body(T["boostingo_taisykle"])

    pdf.h2("2.7. Rankinis algoritmų įgyvendinimas ir jo patikra")
    pdf.body(T["rankinis_igyvendinimas"])
    pdf.code(
        "Newton–Raphson:  β ← β − H⁻¹g,  g = Xᵀ(s·(p−y)) + λβ\n"
        "Boostingas:      g = s·(p−y),  h = s·p·(1−p),  w = −ΣG/(ΣH+λ)\n"
        "Padalijimas:     gain = ½·[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ\n"
        "Miškas:          Gini = 1 − p₁² − p₀²,  p(x) = (1/K)·Σ_k medis_k(x)"
    )
    pdf.body(T["patikros_tekstas"])
    if T["patikros_eilutes"]:
        pdf.table(
            ["Modelis", "Rankinis PR-AUC", "Bibliotekos PR-AUC", "Skirtumas", "Sutapimas"],
            T["patikros_eilutes"],
            [44, 32, 34, 26, 38],
        )
        pdf.caption(
            "8 lentelė. Validavimo imtis. Eksperimente naudojami tik rankiniai modeliai; "
            "bibliotekos čia reikalingos tik kaip nepriklausoma patikra."
        )

    # 3. Rezultatai
    pdf.add_page()
    pdf.h1("3. Rezultatai: metodų palyginimas")
    pdf.body(T["rezultatu_ivadas"])
    pdf.table(
        ["Modelis", "PR-AUC val.", "PR-AUC test.", "Brier", "ECE", "Cost", "t", "FN", "FP"],
        [
            [
                r["model"],
                lt(r["val_pr_auc_for_selection"]),
                lt(r["pr_auc"]),
                lt(r["brier"]),
                lt(r["ece"], 3),
                lt(r["cost"], 0),
                lt(r["threshold"], 3),
                str(int(r["FN"])),
                str(int(r["FP"])),
            ]
            for _, r in summary.iterrows()
        ],
        [36, 20, 20, 16, 14, 16, 12, 12, 12],
    )
    pdf.caption(T["lentele1"])
    pdf.body(T["rezultatu_komentaras"])
    pdf.figure(extra["compare"], "1 pav. PR-AUC ir Brier testo imtyje")
    pdf.figure(PLOTS / "pr_curves.png", "2 pav. Precision–Recall kreivės")

    pdf.add_page()
    pdf.h2("3.1. Kalibracija ir klaidų kaina")
    pdf.body(T["kalibracija"])
    pdf.figure(PLOTS / "calibration.png", "3 pav. Kalibracijos kreivės testo imtyje")
    pdf.figure(PLOTS / "cost_curve.png", T["figura_kaina"])
    pdf.body(T["slenksčio_komentaras"])

    pdf.h1("4. Grupės, klaidos, abliacija ir atsparumas")
    pdf.h2("4.1. Veikimas grupėse")
    sex = groups[groups["group"] == "lytis"]
    pdf.table(
        ["Grupė", "n", "Default dažnis", "PR-AUC", "Brier", "FN dalis", "FP dalis"],
        [
            [
                str(r["value"]),
                str(int(r["n"])),
                lt(r["default_rate"], 3),
                lt(r["pr_auc"]),
                lt(r["brier"]),
                lt(r["fn_rate"], 3),
                lt(r["fp_rate"], 3),
            ]
            for _, r in sex.iterrows()
        ],
        [28, 16, 28, 22, 22, 22, 22],
    )
    pdf.caption(T["lentele2"])
    pdf.figure(PLOTS / "group_prauc.png", "5 pav. PR-AUC pagal lytį ir amžiaus grupę")
    pdf.body(T["grupes"])

    pdf.add_page()
    pdf.h2("4.2. Klaidų analizė")
    pdf.body(T["klaidos"])
    pdf.table(
        ["Klaida", "LIMIT_BAL", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1"],
        [
            [
                str(r["error_type"]),
                lt(r["LIMIT_BAL"], 0),
                lt(r["AGE"], 1),
                lt(r["PAY_1"], 2),
                lt(r["BILL_AMT1"], 0),
                lt(r["PAY_AMT1"], 0),
            ]
            for _, r in profile.iterrows()
        ],
        [22, 32, 22, 22, 38, 38],
    )
    pdf.caption(T["lentele3"])
    pdf.table(
        ["ID", "LIMIT_BAL", "AGE", "PAY_1", "p̂_cal", "tikra y"],
        [
            [str(int(r["ID"])), lt(r["LIMIT_BAL"], 0), str(int(r["AGE"])), str(int(r["PAY_1"])), lt(r["p"], 3), str(int(r["y"]))]
            for _, r in fn.head(5).iterrows()
        ],
        [28, 36, 22, 22, 32, 34],
    )
    pdf.caption("4 lentelė. False negative pavyzdžiai: tikras default, modelis neįspėjo.")
    pdf.table(
        ["ID", "LIMIT_BAL", "AGE", "PAY_1", "p̂_cal", "tikra y"],
        [
            [str(int(r["ID"])), lt(r["LIMIT_BAL"], 0), str(int(r["AGE"])), str(int(r["PAY_1"])), lt(r["p"], 3), str(int(r["y"]))]
            for _, r in fp.head(5).iterrows()
        ],
        [28, 36, 22, 22, 32, 34],
    )
    pdf.caption("5 lentelė. False positive pavyzdžiai: klientas būtų apribotas be pagrindo.")

    pdf.h2("4.3. Abliacija")
    pdf.table(
        ["Variantas", "Pašalinta", "PR-AUC teste", "Brier"],
        [
            [str(r["variant"]), str(r["dropped"]), lt(r["test_pr_auc"]), lt(r["test_brier"])]
            for _, r in ablation.iterrows()
        ],
        [40, 78, 28, 28],
    )
    pdf.caption(T["lentele6"])
    pdf.figure(extra["ablation"], "6 pav. PR-AUC pašalinus požymių grupes")
    pdf.body(T["abliacija"])

    pdf.add_page()
    pdf.h2("4.4. Atsparumas")
    pdf.table(
        ["Scenarijus", "n", "PR-AUC", "Brier", "ECE", "Cost"],
        [
            [
                str(r["scenario"]),
                str(int(r["n"])),
                lt(r["pr_auc"]),
                lt(r["brier"]),
                lt(r["ece"], 3),
                lt(r["cost"], 0),
            ]
            for _, r in robust.iterrows()
        ],
        [62, 18, 24, 22, 22, 26],
    )
    pdf.caption(T["lentele7"])
    pdf.figure(extra["robust"], "7 pav. Atsparumo bandymų PR-AUC")
    pdf.body(T["atsparumas"])

    pdf.h1(T["kodel_lame_antrase"])
    pdf.body(T["kodel_lame"])
    pdf.bullet(T["ribos"])

    pdf.h1("6. Atkuriamumas")
    pdf.body(
        "Programinis kodas ir ataskaita: "
        "https://github.com/Meilaa/Kredito-sipareigojim-nevykdymo-rizikos-prognoz-"
    )
    pdf.body(T["atkuriamumas"])
    pdf.code("python run_experiment.py")
    pdf.body("Rankinių algoritmų patikra su bibliotekomis, svetainė ir Jupyter:")
    pdf.code(
        "python scripts/patikra_rankiniai.py\n"
        "streamlit run svetaine.py\n"
        "jupyter notebook notebooks/egzamino_sprendimas.ipynb"
    )
    pdf.body(
        "Priklausomybės: requirements.txt. Projektas: src/ (duomenys, preprocessing, "
        "rankiniai_modeliai su algoritmais, models mokymui, formulės, metrikos), "
        "artifacts/ (lentelės, grafikai, results.json). Patikrai naudojamas xgboost macOS "
        "aplinkoje gali reikalauti OpenMP: bash scripts/ensure_libomp.sh."
    )

    pdf.h1("7. AI naudojimo žurnalas")
    pdf.body(
        "Cursor agentas naudotas kodo struktūrai, svetainei ir ataskaitai. Citatos, DOI, "
        "duomenų aprašymas ir skaitiniai rezultatai tikrinti atskirai. Sugeneruotas kodas "
        "paleistas lokaliai."
    )
    pdf.h2("Svarbiausios užklausos")
    pdf.bullet(
        [
            "Sukurti galutinio egzamino sprendimą pagal Word užduotį ir kolokviumo planą.",
            "Parodyti sprendimą svetainėje, įskaitant kodą.",
            "Sudėti užduotį, dokumentaciją, grafikus ir lenteles į PDF ir Word.",
            "Parašyti mokymo algoritmus rankiniu būdu, kad jų neatliktų biblioteka.",
        ]
    )
    pdf.h2("Priimta / atmesta")
    pdf.body(T["ai_priimta"])
    pdf.h2("Patikrintos AI klaidos")
    pdf.bullet(T["ai_klaidos"])

    pdf.h1("8. Literatūra")
    pdf.body(
        "[1] Yeh, I.-C., Lien, C.-H. (2009). The comparisons of data mining techniques for the "
        "predictive accuracy of probability of default of credit card clients. Expert Systems "
        "with Applications, 36(2), 2473–2480. DOI: 10.1016/j.eswa.2007.12.020. "
        "UCI DOI: 10.24432/C55S3H."
    )
    pdf.body(
        "[2] Wu, H. (2025). From Scores to Decisions: Comparing Logistic Regression, Random Forest, "
        "and XGBoost for Calibrated, Cost-Sensitive Credit Default Prediction. Advances in "
        "Economics, Management and Political Sciences, 240, 70–79. DOI: 10.54254/2754-1169/2025.BL29296."
    )
    pdf.body(
        "[3] Thi Thu, H. G., Doan, T. V., Ban, H.-B., Quy, T. L. (2026). An experimental study on "
        "fairness-aware machine learning for credit scoring problems. Digital Finance, 8, Article 35. "
        "DOI: 10.1007/s42521-026-00202-6."
    )
    pdf.body(
        "[4] Zhang, C. et al. (2026). Resolving Interpretability Conflicts of Gradient Boosted Decision "
        "Trees (XGBoost) in Credit Risk Assessment. Journal of Organizational and End User Computing, "
        "38(1). DOI: 10.4018/JOEUC.402038."
    )

    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"PDF įrašytas: {path}")
    from make_word_report import build as build_docx

    docx_path = build_docx()
    print(f"Word įrašytas: {docx_path}")
