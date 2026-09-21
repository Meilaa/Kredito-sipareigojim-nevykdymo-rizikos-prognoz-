#!/usr/bin/env python3
"""Sudaro egzamino Word ataskaitą: užduotis, dokumentacija, lentelės, grafikai."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_report import extra_plots  # noqa: E402
from report_texts import lt, surinkti  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
TABLES = ARTIFACTS / "tables"
REPORTS = ROOT / "reports"
OUT = REPORTS / "Egzamino_ataskaita.docx"

NAVY = RGBColor(0x1B, 0x2A, 0x4A)
BURGUNDY = RGBColor(0x8B, 0x1E, 0x3F)
GRAY = RGBColor(0x5A, 0x65, 0x77)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def load_results() -> dict:
    return json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))


def set_run_font(run, name: str = "Calibri", size: int = 11, bold: bool = False, color=NAVY) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = color


def add_text(paragraph, text: str, **kwargs) -> None:
    run = paragraph.add_run(text)
    set_run_font(run, **kwargs)


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    for child in list(tc_pr):
        if child.tag == qn("w:shd"):
            tc_pr.remove(child)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")
    tc_pr.append(shd)


def set_cell_border(cell) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "1B2A4A")
        tc_borders.append(el)
    tc_pr.append(tc_borders)


def set_cell_text(cell, text: str, *, header: bool = False, center: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center or header else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    add_text(
        p,
        text,
        name="Calibri",
        size=9,
        bold=header,
        color=WHITE if header else NAVY,
    )
    shade_cell(cell, "1B2A4A" if header else "FFFFFF")
    set_cell_border(cell)


def body_paragraph(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_text(p, text, size=11)


def heading1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    add_text(p, text, size=16, bold=True, color=NAVY)


def heading2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    add_text(p, text, size=13, bold=True, color=BURGUNDY)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Cm(0.8)
        add_text(p, f"•  {item}", size=11)


def code_block(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.left_indent = Cm(0.3)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "F4F1EC")
    shading.set(qn("w:val"), "clear")
    p._p.get_or_add_pPr().append(shading)
    add_text(p, text, name="Consolas", size=9, color=NAVY)


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(12)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, text, size=9, color=GRAY)


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, header=True, center=True)
    for r_i, row in enumerate(rows, start=1):
        for c_i, val in enumerate(row):
            set_cell_text(table.rows[r_i].cells[c_i], val, center=True)
            if r_i % 2 == 0:
                shade_cell(table.rows[r_i].cells[c_i], "F4F1EC")
    doc.add_paragraph()


def add_figure(doc: Document, path: Path, title: str) -> None:
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    add_text(p, title, size=11, bold=True)
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = pic.add_run()
    run.add_picture(str(path), width=Cm(15.4))


def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)

    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_text(hp, "Intelektualiosios sistemos · galutinis egzaminas · DISfm-26", size=9, color=GRAY)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(fp, "", size=9, color=GRAY)
    run = fp.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)
    set_run_font(run, size=9, color=GRAY)
    return doc


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

    doc = setup_document()

    # Titulinis
    for _ in range(4):
        doc.add_paragraph()
    for line, size, bold in [
        ("VILNIUS TECH", 14, False),
        ("Intelektualiosios sistemos", 13, False),
        ("", 12, False),
        ("Kredito įsipareigojimų nevykdymo rizikos prognozė", 22, True),
        ("Galutinio egzamino ataskaita", 14, False),
        ("", 12, False),
        ("Studentė: Meila Andriuškevičiūtė", 12, False),
        ("Grupė: DISfm-26", 12, False),
        ("2026–2027 m. m.", 12, False),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        if line:
            add_text(p, line, size=size, bold=bold)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(28)
    add_text(
        p,
        "UCI Default of Credit Card Clients · kalibruota default tikimybė · "
        "PR-AUC, Brier score, kalibracija, klaidų kaina",
        size=10,
        color=GRAY,
    )
    doc.add_page_break()

    heading1(doc, "1. Užduotis")
    body_paragraph(
        doc,
        "Individuali 2026–2027 m. užduotis dalykui „Intelektualiosios sistemos“. "
        "Tema: kredito įsipareigojimų nevykdymo rizika. Finansų įstaiga nori numatyti "
        "kito mėnesio įsipareigojimų nevykdymo riziką.",
    )
    heading2(doc, "Duomenys ir sprendimo tikslas")
    bullets(
        doc,
        [
            "Duomenų rinkinys: UCI Default of Credit Card Clients, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "Prognozuojamas rezultatas: kalibruota įsipareigojimų nevykdymo tikimybė.",
            "Privalomas baseline: bendras dažnis ir logistinė regresija.",
            T["intelektualus_metodai"],
            "Pagrindinės vertinimo kryptys: PR-AUC, Brier score, kalibracija ir klaidų kaštų kreivė.",
            "Duomenų skaidymas: stratifikuotas testas; preprocessing tik mokymo imtyje.",
            "Svarbiausias probleminis aspektas: teisingumas skirtingoms grupėms ir tikimybės kalibravimas.",
        ],
    )
    heading2(doc, "Galutinio egzamino reikalavimai")
    bullets(
        doc,
        [
            "Įgyvendinti duomenų paruošimo grandinę, baseline ir bent du intelektualiuosius metodus.",
            "Bent vienam modeliui parodyti formulės ar taisyklės ryšį su konkrečia kodo vieta. "
            "Šiame darbe visi mokymo algoritmai parašyti rankiniu būdu.",
            "Palyginti metodus tuo pačiu skaidymu ir tomis pačiomis metrikomis. Hiperparametrų parinkimui nenaudoti galutinio testo.",
            "Atlikti bent vieną abliaciją ir bent vieną atsparumo bandymą.",
            "Pateikti rezultatų lentelę, grafiką, klaidų pavyzdžius ir paaiškinti, kodėl metodas laimėjo.",
            "Pateikti paleidimo instrukciją, priklausomybes, fiksuotas sėklas ir vieną komandą eksperimentui.",
            "Pateikti AI naudojimo žurnalą.",
        ],
    )

    heading1(doc, "2. Sprendimo dokumentacija")
    heading2(doc, "2.1. Problema")
    body_paragraph(doc, T["problema"])
    heading2(doc, "2.2. Hipotezė")
    body_paragraph(doc, res["hypothesis"])
    body_paragraph(doc, T["hipotezes_isvada"])
    heading2(doc, "2.3. Įvestis ir išvestis")
    body_paragraph(
        doc,
        "Įvestis X: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, mokėjimų būsenos PAY_1–PAY_6, "
        "sąskaitos BILL_AMT1–BILL_AMT6, mokėjimai PAY_AMT1–PAY_AMT6. Tikslas Y: 0 – "
        "įsipareigojimai vykdomi, 1 – kitą mėnesį nevykdomi. Išvestis: p̂ ∈ [0, 1], po "
        "kalibravimo p̂_cal, klasė ŷ = 1, jei p̂_cal ≥ t.",
    )
    heading2(doc, "2.4. Duomenų grandinė")
    body_paragraph(doc, T["duomenu_grandine"])
    code_block(doc, T["grandines_kodas"])
    heading2(doc, "2.5. Modeliai")
    body_paragraph(doc, T["modeliai"])
    heading2(doc, "2.6. Formulės ir kodas")
    body_paragraph(doc, "Logistinė regresija:")
    code_block(doc, "z_i = β0 + β1 x_i1 + … + βp x_ip\np̂_i = 1 / (1 + e^{-z_i})")
    body_paragraph(doc, T["formules_patikra"])
    body_paragraph(doc, "Sprendimo taisyklė, Brier score ir klaidų kaina:")
    code_block(
        doc,
        "ŷ_i = 1, kai p̂_cal,i ≥ t; kitaip ŷ_i = 0\n"
        "BS = (1/N) Σ (p̂_i − y_i)²\n"
        f"Cost(t) = C_FN · FN(t) + C_FP · FP(t),  C_FN = {int(res['c_fn'])}, C_FP = {int(res['c_fp'])}",
    )
    body_paragraph(doc, T["boostingo_taisykle"])

    heading2(doc, "2.7. Rankinis algoritmų įgyvendinimas ir jo patikra")
    body_paragraph(doc, T["rankinis_igyvendinimas"])
    code_block(
        doc,
        "Newton–Raphson:  β ← β − H⁻¹g,  g = Xᵀ(s·(p−y)) + λβ\n"
        "Boostingas:      g = s·(p−y),  h = s·p·(1−p),  w = −ΣG/(ΣH+λ)\n"
        "Padalijimas:     gain = ½·[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ\n"
        "Miškas:          Gini = 1 − p₁² − p₀²,  p(x) = (1/K)·Σ_k medis_k(x)",
    )
    body_paragraph(doc, T["patikros_tekstas"])
    if T["patikros_eilutes"]:
        add_table(
            doc,
            ["Modelis", "Rankinis PR-AUC", "Bibliotekos PR-AUC", "Skirtumas", "Sutapimas"],
            T["patikros_eilutes"],
        )
        caption(
            doc,
            "8 lentelė. Validavimo imtis. Eksperimente naudojami tik rankiniai modeliai; "
            "bibliotekos čia reikalingos tik kaip nepriklausoma patikra.",
        )

    heading1(doc, "3. Rezultatai: metodų palyginimas")
    body_paragraph(doc, T["rezultatu_ivadas"])
    add_table(
        doc,
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
    )
    caption(doc, T["lentele1"])
    body_paragraph(doc, T["rezultatu_komentaras"])
    add_figure(doc, extra["compare"], "1 pav. PR-AUC ir Brier testo imtyje")
    add_figure(doc, PLOTS / "pr_curves.png", "2 pav. Precision–Recall kreivės")

    heading2(doc, "3.1. Kalibracija ir klaidų kaina")
    body_paragraph(doc, T["kalibracija"])
    add_figure(doc, PLOTS / "calibration.png", "3 pav. Kalibracijos kreivės testo imtyje")
    add_figure(doc, PLOTS / "cost_curve.png", T["figura_kaina"])
    body_paragraph(doc, T["slenksčio_komentaras"])

    heading1(doc, "4. Grupės, klaidos, abliacija ir atsparumas")
    heading2(doc, "4.1. Veikimas grupėse")
    sex = groups[groups["group"] == "lytis"]
    add_table(
        doc,
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
    )
    caption(doc, T["lentele2"])
    add_figure(doc, PLOTS / "group_prauc.png", "5 pav. PR-AUC pagal lytį ir amžiaus grupę")
    body_paragraph(doc, T["grupes"])

    heading2(doc, "4.2. Klaidų analizė")
    body_paragraph(doc, T["klaidos"])
    add_table(
        doc,
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
    )
    caption(doc, T["lentele3"])
    add_table(
        doc,
        ["ID", "LIMIT_BAL", "AGE", "PAY_1", "p̂_cal", "tikra y"],
        [
            [str(int(r["ID"])), lt(r["LIMIT_BAL"], 0), str(int(r["AGE"])), str(int(r["PAY_1"])), lt(r["p"], 3), str(int(r["y"]))]
            for _, r in fn.head(5).iterrows()
        ],
    )
    caption(doc, "4 lentelė. False negative pavyzdžiai: tikras default, modelis neįspėjo.")
    add_table(
        doc,
        ["ID", "LIMIT_BAL", "AGE", "PAY_1", "p̂_cal", "tikra y"],
        [
            [str(int(r["ID"])), lt(r["LIMIT_BAL"], 0), str(int(r["AGE"])), str(int(r["PAY_1"])), lt(r["p"], 3), str(int(r["y"]))]
            for _, r in fp.head(5).iterrows()
        ],
    )
    caption(doc, "5 lentelė. False positive pavyzdžiai: klientas būtų apribotas be pagrindo.")

    heading2(doc, "4.3. Abliacija")
    add_table(
        doc,
        ["Variantas", "Pašalinta", "PR-AUC teste", "Brier"],
        [
            [str(r["variant"]), str(r["dropped"]), lt(r["test_pr_auc"]), lt(r["test_brier"])]
            for _, r in ablation.iterrows()
        ],
    )
    caption(doc, T["lentele6"])
    add_figure(doc, extra["ablation"], "6 pav. PR-AUC pašalinus požymių grupes")
    body_paragraph(doc, T["abliacija"])

    heading2(doc, "4.4. Atsparumas")
    add_table(
        doc,
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
    )
    caption(doc, T["lentele7"])
    add_figure(doc, extra["robust"], "7 pav. Atsparumo bandymų PR-AUC")
    body_paragraph(doc, T["atsparumas"])

    heading1(doc, T["kodel_lame_antrase"])
    body_paragraph(doc, T["kodel_lame"])
    bullets(doc, T["ribos"])

    heading1(doc, "6. Atkuriamumas")
    body_paragraph(
        doc,
        "Programinis kodas ir ataskaita: "
        "https://github.com/Meilaa/Kredito-sipareigojim-nevykdymo-rizikos-prognoz-",
    )
    body_paragraph(doc, T["atkuriamumas"])
    code_block(doc, "python run_experiment.py")
    body_paragraph(doc, "Rankinių algoritmų patikra su bibliotekomis, svetainė ir Jupyter:")
    code_block(
        doc,
        "python scripts/patikra_rankiniai.py\n"
        "streamlit run svetaine.py\n"
        "jupyter notebook notebooks/egzamino_sprendimas.ipynb",
    )
    body_paragraph(
        doc,
        "Priklausomybės: requirements.txt. Projektas: src/ (duomenys, preprocessing, "
        "rankiniai_modeliai su algoritmais, models mokymui, formulės, metrikos), "
        "artifacts/ (lentelės, grafikai, results.json). Patikrai naudojamas xgboost macOS "
        "aplinkoje gali reikalauti OpenMP: bash scripts/ensure_libomp.sh.",
    )

    heading1(doc, "7. AI naudojimo žurnalas")
    body_paragraph(
        doc,
        "Cursor agentas naudotas kodo struktūrai, svetainei ir ataskaitai. Citatos, DOI, "
        "duomenų aprašymas ir skaitiniai rezultatai tikrinti atskirai. Sugeneruotas kodas "
        "paleistas lokaliai.",
    )
    heading2(doc, "Svarbiausios užklausos")
    bullets(
        doc,
        [
            "Sukurti galutinio egzamino sprendimą pagal Word užduotį ir kolokviumo planą.",
            "Parodyti sprendimą svetainėje, įskaitant kodą.",
            "Sudėti užduotį, dokumentaciją, grafikus ir lenteles į PDF ir Word.",
            "Parašyti mokymo algoritmus rankiniu būdu, kad jų neatliktų biblioteka.",
        ],
    )
    heading2(doc, "Priimta / atmesta")
    body_paragraph(doc, T["ai_priimta"])
    heading2(doc, "Patikrintos AI klaidos")
    bullets(doc, T["ai_klaidos"])

    heading1(doc, "8. Literatūra")
    body_paragraph(
        doc,
        "[1] Yeh, I.-C., Lien, C.-H. (2009). The comparisons of data mining techniques for the "
        "predictive accuracy of probability of default of credit card clients. Expert Systems "
        "with Applications, 36(2), 2473–2480. DOI: 10.1016/j.eswa.2007.12.020. "
        "UCI DOI: 10.24432/C55S3H.",
    )
    body_paragraph(
        doc,
        "[2] Wu, H. (2025). From Scores to Decisions: Comparing Logistic Regression, Random Forest, "
        "and XGBoost for Calibrated, Cost-Sensitive Credit Default Prediction. Advances in "
        "Economics, Management and Political Sciences, 240, 70–79. DOI: 10.54254/2754-1169/2025.BL29296.",
    )
    body_paragraph(
        doc,
        "[3] Thi Thu, H. G., Doan, T. V., Ban, H.-B., Quy, T. L. (2026). An experimental study on "
        "fairness-aware machine learning for credit scoring problems. Digital Finance, 8, Article 35. "
        "DOI: 10.1007/s42521-026-00202-6.",
    )
    body_paragraph(
        doc,
        "[4] Zhang, C. et al. (2026). Resolving Interpretability Conflicts of Gradient Boosted Decision "
        "Trees (XGBoost) in Credit Risk Assessment. Journal of Organizational and End User Computing, "
        "38(1). DOI: 10.4018/JOEUC.402038.",
    )

    doc.save(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Word įrašytas: {path}")
