#!/usr/bin/env python3
"""Sudaro egzamino PDF ataskaitą: užduotis, dokumentacija, lentelės, grafikai."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
TABLES = ARTIFACTS / "tables"
REPORTS = ROOT / "reports"
OUT = REPORTS / "Egzamino_ataskaita.pdf"
FONT_DIR = ROOT / ".venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf"


def lt(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}".replace(",", " ")
    text = f"{float(value):.{digits}f}"
    return text.replace(".", ",")


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
    ax.set_title("Abliacija: XGBoost be požymių grupių")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["ablation"] = PLOTS / "report_ablation.png"
    fig.savefig(paths["ablation"], dpi=170)
    plt.close(fig)

    rob = pd.read_csv(TABLES / "robustness.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(rob["scenario"], rob["pr_auc"], color="#4C6B8A")
    ax.set_xlabel("PR-AUC")
    ax.set_title("Atsparumo bandymai (XGBoost, kalibruotos tikimybės)")
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
            "Svarstytini intelektualieji metodai: SVM, MLP, medžių ansamblis, fuzzy taisyklės. Šiame darbe pasirinkti XGBoost ir Random Forest.",
            "Pagrindinės vertinimo kryptys: PR-AUC, Brier score, kalibracija ir klaidų kaštų kreivė.",
            "Duomenų skaidymas: stratifikuotas testas; preprocessing tik mokymo imtyje.",
            "Svarbiausias probleminis aspektas: teisingumas skirtingoms grupėms ir tikimybės kalibravimas.",
        ]
    )
    pdf.h2("Galutinio egzamino reikalavimai")
    pdf.bullet(
        [
            "Įgyvendinti duomenų paruošimo grandinę, baseline ir bent du intelektualiuosius metodus.",
            "Bent vienam modeliui parodyti formulės ar taisyklės ryšį su konkrečia kodo vieta.",
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
    pdf.body(
        "Tikslas – pagal kredito kortelės kliento istorinius duomenis įvertinti kalibruotą "
        "tikimybę, kad klientas kitą mėnesį nevykdys finansinių įsipareigojimų. Modelis "
        "vertinamas kaip sprendimų palaikymo priemonė, ne automatinė galutinio kredito "
        "sprendimo sistema. Rinkinį sudaro 30 000 klientų, iš jų 6 636 (22,12 %) priklauso "
        "default klasei."
    )
    pdf.h2("2.2. Hipotezė")
    pdf.body(res["hypothesis"])
    pdf.body(
        "Išvada: hipotezė pasitvirtino. Testo imtyje XGBoost pasiekė PR-AUC 0,5430, "
        "Random Forest – 0,5333, logistinė regresija – 0,4795, bendras dažnis – 0,2211. "
        "Po isotonic kalibravimo XGBoost Brier score yra 0,1343, ECE – 0,0176."
    )
    pdf.h2("2.3. Įvestis ir išvestis")
    pdf.body(
        "Įvestis X: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, mokėjimų būsenos PAY_1–PAY_6, "
        "sąskaitos BILL_AMT1–BILL_AMT6, mokėjimai PAY_AMT1–PAY_AMT6. Tikslas Y: 0 – "
        "įsipareigojimai vykdomi, 1 – kitą mėnesį nevykdomi. Išvestis: p̂ ∈ [0, 1], po "
        "kalibravimo p̂_cal, klasė ŷ = 1, jei p̂_cal ≥ t."
    )
    pdf.h2("2.4. Duomenų grandinė")
    pdf.body(
        "Stratifikuotas 70/15/15 skaidymas, sėkla 42. Mokymas 21 000, validavimas 4 500, "
        "testas 4 500. EDUCATION reikšmės 0, 5, 6 sujungtos į 4; MARRIAGE 0 – į 3. "
        "Kategorijos koduojamos OneHotEncoder, skaitiniai požymiai logistinei regresijai "
        "standartizuojami. Visi preprocessing parametrai skaičiuojami tik mokymo imtyje."
    )
    pdf.code(
        "UCI duomenys → kokybės patikra → stratifikuotas 70/15/15 →\n"
        "preprocessing tik pagal mokymą → bendras dažnis + logistinė regresija →\n"
        "XGBoost + Random Forest → validavimo metrikos → kalibravimas →\n"
        "slenksčio / kainos analizė → grupės → abliacija / atsparumas → vienkartinis testas"
    )
    pdf.h2("2.5. Modeliai")
    pdf.body(
        "Naivus baseline visiems klientams priskiria mokymo imties default dažnį (0,2212). "
        "Logistinė regresija yra interpretuojamas baseline; hiperparametras C ∈ {0,1; 1; 10} "
        "ir class_weight parinkti pagal validavimo PR-AUC. XGBoost – pagrindinis intelektualus "
        "metodas (n_estimators=150, max_depth=3, learning_rate=0,05). Random Forest – antroji "
        "alternatyva (200 medžių, max_depth=8). Kalibratorius (isotonic arba Platt) mokomas "
        "tik validavime; visiems modeliams pasirinktas isotonic."
    )
    pdf.h2("2.6. Formulės ir kodas")
    pdf.body("Logistinė regresija:")
    pdf.code("z_i = β0 + β1 x_i1 + … + βp x_ip\np̂_i = 1 / (1 + e^{-z_i})")
    pdf.body(
        "Šios formulės įgyvendintos faile src/formulas.py funkcijose linear_score ir "
        "logistic_probability. Validavimo imties aštuoniose eilutėse rankinės tikimybės "
        f"sutampa su sklearn iki {res['formula_check']['max_abs_diff']:.2e}."
    )
    pdf.body("Sprendimo taisyklė, Brier score ir klaidų kaina:")
    pdf.code(
        "ŷ_i = 1, kai p̂_cal,i ≥ t; kitaip ŷ_i = 0\n"
        "BS = (1/N) Σ (p̂_i − y_i)²\n"
        "Cost(t) = C_FN · FN(t) + C_FP · FP(t),  C_FN = 5, C_FP = 1"
    )
    pdf.body(
        "XGBoost dvejetainei klasifikacijai naudoja objective='binary:logistic': "
        "z(x) = Σ_k f_k(x), p̂ = σ(z). Slenkstis t parenkamas validavime minimizuojant Cost(t)."
    )

    # 3. Rezultatai
    pdf.add_page()
    pdf.h1("3. Rezultatai: metodų palyginimas")
    pdf.body(
        "Visi metodai lyginti tomis pačiomis imtimis. Hiperparametrai, kalibratorius ir "
        "slenkstis parinkti nenaudojant testo. Pagrindinė diskriminacijos metrika – PR-AUC, "
        "nes klasės nesubalansuotos. Accuracy čia klaidina: pastovus „nėra default“ duotų apie 78 %."
    )
    pdf.table(
        ["Modelis", "PR-AUC", "ROC-AUC", "Brier", "ECE", "Cost", "t", "FN", "FP"],
        [
            [
                r["model"],
                lt(r["pr_auc"]),
                lt(r["roc_auc"]),
                lt(r["brier"]),
                lt(r["ece"], 3),
                lt(r["cost"], 0),
                lt(r["threshold"], 3),
                str(int(r["FN"])),
                str(int(r["FP"])),
            ]
            for _, r in summary.iterrows()
        ],
        [36, 16, 18, 16, 14, 16, 14, 12, 12],
    )
    pdf.caption(
        "1 lentelė. Kalibruotos tikimybės, testo imtis (n = 4 500). Mažesnis Brier ir ECE – geriau."
    )
    pdf.body(
        "XGBoost laimėjo pagal PR-AUC. Random Forest arti jo pagal diskriminaciją, bet "
        "turi daugiau FP (1 475 prieš 924) ir žemesnį tikslumą prie savo slenksčio. "
        "Logistinė regresija silpnesnė ir pagal PR-AUC, ir pagal Brier. Bendras dažnis "
        "PR-AUC sutampa su default dažniu – tai naivi atskaitos riba."
    )
    pdf.figure(extra["compare"], "1 pav. PR-AUC ir Brier testo imtyje")
    pdf.figure(PLOTS / "pr_curves.png", "2 pav. Precision–Recall kreivės")

    pdf.add_page()
    pdf.h2("3.1. Kalibracija ir klaidų kaina")
    pdf.body(
        "Geras surikiavimas pagal riziką nereiškia, kad prognozuota 0,30 tikimybė atitinka "
        "30 % įvykių. Todėl tikimybės kalibruotos isotonic regresija validavimo imtyje. "
        "XGBoost Brier po kalibravimo sumažėjo nuo 0,175 iki 0,130 (validavime)."
    )
    pdf.figure(PLOTS / "calibration.png", "3 pav. Kalibracijos kreivės testo imtyje")
    pdf.figure(PLOTS / "cost_curve.png", "4 pav. XGBoost klaidų kainos kreivė; pasirinktas t = 0,175")
    pdf.body(
        "Slenkstis 0,175 pasirinktas ne pagal accuracy, o pagal Cost(t) su C_FN:C_FP = 5:1. "
        "Žemesnis slenkstis padidintų recall, bet ir nepagrįstų ribojimų (FP) skaičių."
    )

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
    pdf.caption("2 lentelė. XGBoost pagal lytį (testas). PR-AUC panašus (0,541 ir 0,546).")
    pdf.figure(PLOTS / "group_prauc.png", "5 pav. PR-AUC pagal lytį ir amžiaus grupę")
    pdf.body(
        "Amžiaus grupėje 30–39 PR-AUC žemesnis (0,493) nei 50+ (0,590). Skirtumai "
        "neįrodo priežastinės diskriminacijos, bet demografiniai požymiai gali būti proxy. "
        "Abliacija be demografijos PR-AUC sumažina nedaug (0,562 → 0,553, nekalibruotas "
        "XGBoost palyginimas), todėl praktikoje demografiją galima šalinti be didelio nuostolio."
    )

    pdf.add_page()
    pdf.h2("4.2. Klaidų analizė")
    pdf.body(
        "False negative (praleistas default) brangesnis. FN klientai vidutiniškai turi "
        "didesnį limitą ir geresnę PAY_1 būseną nei TP – modelis juos laiko saugesniais. "
        "FP klientai dažniau turi teigiamą PAY_1 (vėlavimą) ir mažesnį limitą."
    )
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
    pdf.caption("3 lentelė. Vidutiniai požymiai pagal TP/TN/FP/FN (testas, XGBoost, t = 0,175).")
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
    pdf.caption("6 lentelė. XGBoost abliacija. Ši lentelė naudoja nekalibruotas tikimybes, kad lyginti požymių poveikį.")
    pdf.figure(extra["ablation"], "6 pav. PR-AUC pašalinus požymių grupes")
    pdf.body(
        "Mokėjimų būsenos yra svarbiausia grupė: be PAY_* PR-AUC krenta iki 0,446. "
        "Demografijos pašalinimas pakeičia rezultatą nedaug."
    )

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
    pdf.caption("7 lentelė. Tas pats XGBoost modelis, be pakartotinio mokymo.")
    pdf.figure(extra["robust"], "7 pav. Atsparumo bandymų PR-AUC")
    pdf.body(
        "Gauso triukšmas (σ = 0,15 · std) PR-AUC sumažina nuo 0,543 iki 0,487. "
        "10 % atsitiktinai trūkstamų reikšmių poveikis mažesnis (0,527). "
        "Amžiaus grupėje ≥ 50 diskriminacija išlieka, bet imtis mažesnė (n = 422)."
    )

    pdf.h1("5. Kodėl XGBoost laimėjo ir kokios ribos")
    pdf.body(
        "XGBoost pranoko logistinę regresiją, nes kredito vėlavimo istorija ir mokėjimų "
        "sumos sąveikauja netiesiškai – tai medžiams natūralu, o logistinei regresijai "
        "reikėtų rankinių narių. Random Forest artimas, bet gradientinis klaidų taisymas "
        "čia davė nedidelį PR-AUC pranašumą. MLP nebuvo treniruotas šiame paleidime: "
        "du intelektualūs metodai tenkina užduotį, o MLP jautresnis masteliui."
    )
    pdf.bullet(
        [
            "Duomenys surinkti Taivane 2005 m. – rezultatų negalima perkelti į dabartinę Lietuvos rinką.",
            "Modelis neduoda priežastinio paaiškinimo ir neturi būti vienintelis kredito sprendimas.",
            "Slenkstis priklauso nuo C_FN ir C_FP; kitos kainos duotų kitą t.",
            "Isotonic kalibravimas p̂_cal keičia laiptais – panašūs balai gali gauti tą pačią tikimybę.",
            "Klasių disbalansas (~22 %) daro accuracy klaidinančia metrika.",
        ]
    )

    pdf.h1("6. Atkuriamumas")
    pdf.body(
        "Programinis kodas ir ataskaita: "
        "https://github.com/Meilaa/Kredito-sipareigojim-nevykdymo-rizikos-prognoz-"
    )
    pdf.body("Atsitiktinė sėkla 42. Duomenys atsisiunčiami automatiniu būdu iš UCI (id=350). Viena komanda:")
    pdf.code("python run_experiment.py")
    pdf.body("Svetainė ir Jupyter:")
    pdf.code("streamlit run svetaine.py\njupyter notebook notebooks/egzamino_sprendimas.ipynb")
    pdf.body(
        "Priklausomybės: requirements.txt. Projektas: src/ (duomenys, preprocessing, modeliai, "
        "formulės, metrikos), artifacts/ (lentelės, grafikai, results.json). macOS XGBoost "
        "gali reikalauti OpenMP: bash scripts/ensure_libomp.sh."
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
            "Sudėti užduotį, dokumentaciją, grafikus ir lenteles į PDF.",
        ]
    )
    pdf.h2("Priimta / atmesta")
    pdf.body(
        "Priimta: modulinė Python grandinė, XGBoost ir Random Forest, 70/15/15, Streamlit, "
        "formulės src/formulas.py. Atmesta: tik Jupyter be svetainės; MLP kaip trečias metodas "
        "šiame paleidime."
    )
    pdf.h2("Patikrintos AI klaidos")
    pdf.bullet(
        [
            "ucimlrepo grąžino stulpelius X1–X23, ne LIMIT_BAL. Pirmas mokymas žlugo su 0 požymių. Pataisyta UCI_X_MAP, patikrinta pagal Yeh ir Lien aprašą.",
            "pip install xgboost macOS nepaliko libomp.dylib. Patikra: import XGBClassifier po libomp įrašymo.",
            "Straipsnio [2] PR-AUC ≈ 0,89 neperkeltina į šį UCI rinkinį – DOI patikrintas, metrikos imamos tik iš run_experiment.py.",
        ]
    )

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
