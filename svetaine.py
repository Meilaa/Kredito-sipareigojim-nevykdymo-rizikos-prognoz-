#!/usr/bin/env python3
"""Gynimo svetainė: paleidimas `streamlit run svetaine.py`."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.formulas import (
    confusion_counts,
    decision_from_threshold,
    total_error_cost,
)
from src.metrics import select_threshold_by_cost

RESULTS_PATH = ARTIFACTS_DIR / "results.json"
SRC_DIR = PROJECT_ROOT / "src"


def load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def load_table(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_source(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return f"{float(value):.{digits}f}"


def inject_css() -> None:
    st.markdown(
        "<style>.block-container {max-width: 1180px; padding-top: 1.1rem;}</style>",
        unsafe_allow_html=True,
    )


def page_home(results: dict) -> None:
    st.title("Kredito įsipareigojimų nevykdymo rizikos prognozė")
    st.caption("Meila Andriuškevičiūtė · DISfm-26 · Intelektualiosios sistemos · galutinis egzaminas")
    winner = results["winner"]
    row = next(r for r in results["summary"] if r["model"] == winner)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Laimėtojas pagal PR-AUC", winner)
    c2.metric("PR-AUC (testas)", fmt(row["pr_auc"]))
    c3.metric("Brier score", fmt(row["brier"]))
    c4.metric("Slenkstis t", fmt(row["threshold"], 2))

    st.subheader("Ką šis sprendimas daro")
    st.write(
        "Pagal kredito kortelės kliento istoriją modelis įvertina **kalibruotą tikimybę**, "
        "kad kitą mėnesį nebus vykdomi finansiniai įsipareigojimai. Tai sprendimų palaikymo "
        "priemonė, ne automatinis kredito verdiktas."
    )
    st.info(
        f"**Hipotezė.** {results['hypothesis']}\n\n"
        f"**Išvada.** Testo imtyje pagal PR-AUC laimėjo **{winner}** "
        f"(PR-AUC = {fmt(row['pr_auc'])}, Brier = {fmt(row['brier'])}). "
        "Hiperparametrai, kalibravimas ir slenkstis parinkti tik validavimo imtyje."
    )


def page_problem(results: dict) -> None:
    st.header("Problema, duomenys ir grandinė")
    q = results["quality"]
    s = results["split"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Klientų", f"{q['n_rows']:,}".replace(",", " "))
    c2.metric("Default klasė", f"{q['n_default']:,}".replace(",", " "))
    c3.metric("Default dažnis", f"{q['default_rate']*100:.2f} %")
    c4.metric("Trūkstamos reikšmės", str(sum(q["missing_by_column"].values())))

    st.markdown(
        """
        **Įvestis X.** `LIMIT_BAL`, demografija (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`),
        6 mėn. mokėjimų būsenos (`PAY_1`–`PAY_6`), sąskaitos (`BILL_AMT1`–`6`) ir mokėjimai (`PAY_AMT1`–`6`).

        **Išvestis.** `p̂ ∈ [0,1]` – default tikimybė; po kalibravimo `p̂_cal`;
        klasė `ŷ = 1`, jei `p̂_cal ≥ t`.
        """
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Skaidymas")
        st.dataframe(
            pd.DataFrame(
                [
                    {"imtis": "mokymas", "n": s["n_train"], "default dažnis": s["train_default_rate"]},
                    {"imtis": "validavimas", "n": s["n_val"], "default dažnis": s["val_default_rate"]},
                    {"imtis": "testas", "n": s["n_test"], "default dažnis": s["test_default_rate"]},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(s["note"])
    with col_b:
        st.subheader("Kodėl ne vien accuracy")
        st.write(
            "Default klasė sudaro apie 22 %. Modelis, visada sakantis „ne default“, "
            "turėtų ~78 % accuracy, bet būtų nenaudingas rizikai. Todėl pagrindinės "
            "metrikos yra **PR-AUC**, **Brier score**, kalibracijos kreivė ir klaidų kaina."
        )
    st.subheader("Algoritmo eiga")
    st.code(
        "UCI duomenys → kokybės patikra → stratifikuotas 70/15/15 →\n"
        "preprocessing tik pagal mokymą → bendras dažnis + logistinė regresija →\n"
        "XGBoost + Random Forest → validavimo metrikos → kalibravimas →\n"
        "slenksčio / kainos analizė → grupės → abliacija / atsparumas → vienkartinis testas",
        language="text",
    )
    st.subheader("Duomenų paruošimo kodas")
    st.code(read_source("src/preprocess.py"), language="python")


def page_formulas(results: dict) -> None:
    st.header("Formulės ir jų ryšys su kodu")
    st.info(
        "Logistinės regresijos tikimybė skaičiuojama rankine formule faile `src/formulas.py` "
        f"ir sutampa su sklearn. Skirtumas šiame eksperimente: "
        f"**{results['formula_check']['max_abs_diff']:.2e}**."
    )

    st.subheader("1. Logistinė regresija")
    st.latex(r"z_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}")
    st.latex(r"\hat{p}_i = \frac{1}{1 + e^{-z_i}}")
    st.write(
        "`xᵢⱼ` – i-ojo kliento j-asis požymis po preprocessing; `β₀` – laisvasis narys "
        f"({fmt(results['logistic_extra']['intercept'])}); `βⱼ` – mokymo metu gauti koeficientai; "
        "`p̂ᵢ` – default tikimybė."
    )
    src = read_source("src/formulas.py")
    start = src.index("def linear_score")
    end = src.index("def decision_from_threshold")
    st.code(src[start:end], language="python")

    st.subheader("2. Sprendimo taisyklė")
    st.latex(r"\hat{y}_i = 1 \text{, kai } \hat{p}_{\mathrm{cal},i} \ge t;\quad \text{kitu atveju } \hat{y}_i = 0")
    start = src.index("def decision_from_threshold")
    end = src.index("def brier_score")
    st.code(src[start:end], language="python")

    st.subheader("3. Brier score")
    st.latex(r"\mathrm{BS} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2")
    start = src.index("def brier_score")
    end = src.index("def confusion_counts")
    st.code(src[start:end], language="python")

    st.subheader("4. Klaidų kaina")
    st.latex(r"\mathrm{Cost}(t) = C_{\mathrm{FN}}\cdot FN(t) + C_{\mathrm{FP}}\cdot FP(t)")
    start = src.index("def total_error_cost")
    end = src.index("def expected_calibration_error")
    st.code(src[start:end], language="python")

    st.subheader("Skaitiška patikra: formulė vs sklearn")
    check = results["formula_check"]
    demo = pd.DataFrame(
        {
            "z": check["z"],
            "p_formule": check["p_formula"],
            "p_sklearn": check["p_sklearn"],
        }
    )
    st.dataframe(demo, hide_index=True, width="stretch")
    st.caption("Sutapimas rodo, kad tikimybė yra sigmoidinė transformacija σ(z), o ne atskiras neskaičiuojamas žingsnis.")

    st.subheader("XGBoost taisyklė")
    st.write(
        "Dvejetainei klasifikacijai su `objective='binary:logistic'` galutinė tikimybė yra "
        "ta pati sigmoidinė transformacija, tik `z` gaunamas kaip medžių suma: "
        r"$z(x) = \sum_k f_k(x)$, $p̂ = \sigma(z)$."
    )
    st.code(read_source("src/models.py").split("def train_xgboost")[1].split("def logistic_formula_check")[0], language="python")


def page_models(results: dict) -> None:
    st.header("Baseline ir intelektualieji metodai")
    st.write("Visi modeliai mokyti tomis pačiomis imtimis. Hiperparametrai rinkti pagal validavimo PR-AUC.")
    summary = pd.DataFrame(results["summary"])
    show = summary[
        ["model", "kind", "params", "calibration", "threshold", "pr_auc", "brier", "ece", "cost"]
    ]
    st.dataframe(show, hide_index=True, width="stretch")
    st.markdown(
        """
        | Metodas | Vaidmuo | Kodėl tinka |
        |---|---|---|
        | Bendras dažnis | naivus baseline | parodo minimalią atskaitos ribą |
        | Logistinė regresija | interpretuojamas baseline | natūrali tikimybė `σ(z)` |
        | XGBoost | pagrindinis intelektualus | netiesiškumas ir požymių sąveikos |
        | Random Forest | alternatyva | kitas medžių ansamblis, be gradientinio klaidų taisymo |
        """
    )
    st.subheader("Modelių mokymo kodas")
    file = st.selectbox("Failas", ["src/models.py", "src/calibrate.py", "src/experiment.py"])
    st.code(read_source(file), language="python")


def _bar_figure(summary: pd.DataFrame, metric: str, title: str, y_title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=summary["model"],
            y=summary[metric],
            marker_color=["#8B1E3F" if m == summary.iloc[0]["model"] else "#1B2A4A" for m in summary["model"]],
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=50, b=40, l=40, r=20),
        height=380,
    )
    return fig


def page_results(results: dict) -> None:
    st.header("Rezultatai: vienodas palyginimas")
    summary = pd.DataFrame(results["summary"]).sort_values("pr_auc", ascending=False)
    winner = results["winner"]
    st.write(
        f"**{winner}** laimėjo pagal PR-AUC testo imtyje. Tai nereiškia, kad jis geriausias visomis "
        "prasmėmis: Brier ir ECE rodo tikimybių kokybę, o `cost` – verslo kainą prie pasirinkto slenksčio."
    )
    st.dataframe(summary.drop(columns=["params"], errors="ignore"), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _bar_figure(summary, "pr_auc", "PR-AUC testo imtyje (didesnis geriau)", "PR-AUC"),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            _bar_figure(summary.sort_values("brier"), "brier", "Brier score (mažesnis geriau)", "Brier"),
            width="stretch",
        )

    pr_fig = go.Figure()
    for row in results["summary"]:
        name = row["model"]
        ev = results["test_eval"][name]["calibrated"]
        # PR arrays were stripped from JSON; load from plot or recompute from predictions if available
    preds = load_table("test_predictions.csv")
    if not preds.empty:
        from sklearn.metrics import precision_recall_curve

        y = preds["y"].to_numpy()
        for col in preds.columns:
            if not col.startswith("p_cal_"):
                continue
            name = col.replace("p_cal_", "")
            precision, recall, _ = precision_recall_curve(y, preds[col])
            pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
        pr_fig.update_layout(
            title="Precision–Recall kreivės (kalibruotos tikimybės, testas)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=420,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(pr_fig, width="stretch")
    st.caption("Šaltinis: UCI Default of Credit Card Clients, stratifikuotas 15 % testas, sėkla 42.")
    plots = ARTIFACTS_DIR / "plots"
    if (plots / "pr_curves.png").exists():
        st.image(str(plots / "pr_curves.png"), caption="PR kreivės (matplotlib, artifacts/plots)")
        st.image(str(plots / "model_bars.png"), caption="PR-AUC ir Brier palyginimas")


def page_calibration(results: dict) -> None:
    st.header("Kalibracija, slenkstis ir klaidų kaina")
    st.write(
        f"Kalibratorius mokomas **tik validavimo imtyje**. Testas naudojamas vieną kartą. "
        f"Numatytos kainos: C_FN = {results['c_fn']} (praleistas default), C_FP = {results['c_fp']} "
        "(nepagrįstas ribojimas)."
    )
    st.dataframe(pd.DataFrame(results["calibration_choice"]).T.reset_index().rename(columns={"index": "model"}), hide_index=True, width="stretch")

    winner = results["winner"]
    curve = load_table(f"cost_curve_{winner.replace(' ', '_')}.csv")
    if not curve.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["cost"], mode="lines", name="Cost(t)"))
        fig.add_vline(x=results["thresholds"][winner], line_dash="dash", line_color="#8B1E3F")
        fig.update_layout(
            title=f"Klaidų kainos kreivė: {winner}",
            xaxis_title="Slenkstis t",
            yaxis_title="Cost(t) = C_FN·FN + C_FP·FP",
            height=420,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")

    cal_name = f"calibration_{winner.replace(' ', '_')}.csv"
    cal = load_table(cal_name)
    if not cal.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="ideali", line=dict(dash="dash", color="#888")))
        fig.add_trace(
            go.Scatter(
                x=cal["mean_predicted"],
                y=cal["mean_observed"],
                mode="lines+markers",
                name=winner,
            )
        )
        fig.update_layout(
            title="Kalibracijos kreivė testo imtyje",
            xaxis_title="Prognozuota tikimybė",
            yaxis_title="Stebėtas default dažnis",
            height=420,
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Klaidų kainų poveikis slenksčiui")
    st.caption("Slenkstis parenkamas validavimo imtyje pagal Cost(t). Modelio iš naujo mokyti nereikia.")
    c_fn = st.slider("C_FN (praleisto default kaina)", 1.0, 20.0, float(results["c_fn"]), 1.0)
    c_fp = st.slider("C_FP (klaidingo įtarimo kaina)", 1.0, 10.0, float(results["c_fp"]), 1.0)
    val = load_table("val_predictions.csv")
    test = load_table("test_predictions.csv")
    col = f"p_cal_{winner}"
    if not val.empty and col in val.columns:
        t_new, _ = select_threshold_by_cost(val["y"].to_numpy(), val[col].to_numpy(), c_fn, c_fp)
        y_pred = decision_from_threshold(test[col].to_numpy(), t_new)
        counts = confusion_counts(test["y"].to_numpy(), y_pred)
        cost = total_error_cost(test["y"].to_numpy(), y_pred, c_fn, c_fp)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Slenkstis t", f"{t_new:.2f}")
        k2.metric("FN teste", counts["FN"])
        k3.metric("FP teste", counts["FP"])
        k4.metric("Cost(t)", f"{cost:,.0f}")


def page_groups(results: dict) -> None:
    st.header("Teisingumas grupėse ir klaidų pavyzdžiai")
    groups = pd.DataFrame(results["groups"])
    choice = st.selectbox("Grupė", sorted(groups["group"].unique()))
    part = groups[groups["group"] == choice]
    st.dataframe(part, hide_index=True, width="stretch")
    fig = go.Figure(go.Bar(x=part["value"], y=part["pr_auc"], marker_color="#1B2A4A"))
    fig.update_layout(
        title=f"PR-AUC pagal grupę: {choice}",
        yaxis_title="PR-AUC",
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")
    st.write(
        "Demografiniai požymiai gali būti jautrių savybių proxy. Skirtumai tarp grupių "
        "nereiškia priežastinio diskriminavimo. Modelį būtų rizikinga naudoti be atskiros "
        "atitikties peržiūros."
    )
    st.subheader("Klaidų profilis")
    st.dataframe(pd.DataFrame(results["error_profile"]), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("False negative pavyzdžiai")
        st.caption("Tikras default, modelis neįspėjo – potencialus kredito nuostolis.")
        st.dataframe(pd.DataFrame(results["fn_examples"]), hide_index=True, width="stretch")
    with c2:
        st.subheader("False positive pavyzdžiai")
        st.caption("Klientas būtų apribotas be pagrindo.")
        st.dataframe(pd.DataFrame(results["fp_examples"]), hide_index=True, width="stretch")


def page_ablation(results: dict) -> None:
    st.header("Abliacija ir atsparumas")
    st.subheader("Požymių grupių pašalinimas")
    ab = pd.DataFrame(results["ablation"])
    st.dataframe(ab, hide_index=True, width="stretch")
    fig = go.Figure(go.Bar(x=ab["variant"], y=ab["test_pr_auc"], marker_color="#1B2A4A"))
    fig.update_layout(
        title="XGBoost PR-AUC pašalinus požymių grupes",
        yaxis_title="PR-AUC teste",
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")
    st.subheader("Atsparumo bandymai")
    rob = pd.DataFrame(results["robustness"])
    st.dataframe(rob, hide_index=True, width="stretch")
    st.write(
        "Triukšmas ir 10 % trūkstamų reikšmių tikrinami **jau apmokytame** modelyje, "
        "neperrenkant hiperparametrų. Tai parodo, kiek prognozė keičiasi, jei įvestis tampa netiksli."
    )


def _load_bundle(model_name: str):
    import joblib

    path = MODELS_DIR / f"{model_name.replace(' ', '_')}.joblib"
    return joblib.load(path)


PRESETS = {
    "Patikimas klientas": {
        "limit": 360000,
        "age": 34,
        "sex": 2,
        "edu": 2,
        "mar": 2,
        "pay1": -1,
        "pay2": -1,
        "pay3": -1,
        "bill1": 5000,
        "payamt1": 5000,
    },
    "Vėluojantis klientas": {
        "limit": 20000,
        "age": 45,
        "sex": 1,
        "edu": 3,
        "mar": 1,
        "pay1": 2,
        "pay2": 2,
        "pay3": 2,
        "bill1": 18000,
        "payamt1": 0,
    },
}


def _init_live_state() -> None:
    defaults = {
        "limit": 120000,
        "age": 35,
        "sex": 2,
        "edu": 2,
        "mar": 1,
        "pay1": 0,
        "pay2": 0,
        "pay3": 0,
        "bill1": 40000,
        "payamt1": 2000,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def page_live(results: dict) -> None:
    import joblib

    st.header("Gyvas bandymas")
    st.write(
        "Tikimybė perskaičiuojama keičiant laukus. Svarbiausi požymiai – mokėjimų vėlavimas "
        "(`PAY_1`–`PAY_3`) ir kredito limitas. Kalibruota tikimybė kinta laiptais, nes isotonic "
        "kalibratorius sujungia panašius balus į tuos pačius intervalus; nekalibruota `p̂` kinta nuolat."
    )
    model_name = st.selectbox("Modelis", [row["model"] for row in results["summary"]])
    bundle = _load_bundle(model_name)
    model = bundle["model"]
    calibrator = bundle["calibrator"]
    default_t = float(bundle["threshold"])
    t = st.slider("Slenkstis t", 0.05, 0.95, default_t, 0.01)

    tab_one, tab_csv = st.tabs(["Vienas klientas", "CSV failas"])
    with tab_one:
        _init_live_state()
        b1, b2 = st.columns(2)
        if b1.button("Patikimas klientas"):
            st.session_state.update(PRESETS["Patikimas klientas"])
            st.rerun()
        if b2.button("Vėluojantis klientas"):
            st.session_state.update(PRESETS["Vėluojantis klientas"])
            st.rerun()

        c1, c2, c3 = st.columns(3)
        limit = c1.number_input("LIMIT_BAL", 10000, 1000000, step=10000, key="limit")
        age = c2.number_input("AGE", 21, 80, step=1, key="age")
        sex = c3.selectbox("SEX", [1, 2], format_func=lambda x: "vyras" if x == 1 else "moteris", key="sex")
        c1, c2, c3 = st.columns(3)
        edu = c1.selectbox("EDUCATION", [1, 2, 3, 4], key="edu")
        mar = c2.selectbox("MARRIAGE", [1, 2, 3], key="mar")
        pay1 = c3.number_input("PAY_1 (paskutinis mėnuo)", -2, 8, step=1, key="pay1")
        c1, c2, c3 = st.columns(3)
        pay2 = c1.number_input("PAY_2", -2, 8, step=1, key="pay2")
        pay3 = c2.number_input("PAY_3", -2, 8, step=1, key="pay3")
        payamt1 = c3.number_input("PAY_AMT1", 0, 500000, step=500, key="payamt1")
        bill1 = st.number_input("BILL_AMT1", -200000, 800000, step=1000, key="bill1")

        schema = joblib.load(MODELS_DIR / "schema_example.joblib")
        row = {col: 0 for col in schema.columns}
        row.update(
            {
                "LIMIT_BAL": limit,
                "AGE": age,
                "SEX": sex,
                "EDUCATION": edu,
                "MARRIAGE": mar,
                "PAY_1": pay1,
                "PAY_2": pay2,
                "PAY_3": pay3,
                "PAY_4": pay3,
                "PAY_5": pay3,
                "PAY_6": pay3,
                "BILL_AMT1": bill1,
                "BILL_AMT2": bill1,
                "PAY_AMT1": payamt1,
                "PAY_AMT2": payamt1,
            }
        )
        X = pd.DataFrame([row])[list(schema.columns)]
        p_raw = float(model.predict_proba(X)[0])
        p_cal = float(calibrator.transform(np.array([p_raw]))[0])
        yhat = int(p_cal >= t)
        k1, k2, k3 = st.columns(3)
        k1.metric("p̂ (nekalibruota)", f"{p_raw:.3f}")
        k2.metric("p̂_cal", f"{p_cal:.3f}")
        k3.metric("ŷ", "default rizika" if yhat == 1 else "įsipareigojimai vykdomi")
        st.caption(
            f"Sprendimas: ŷ = 1, jei p̂_cal ≥ {t:.2f}. "
            "Jei keičiate tik amžių ar lytį, kalibruota tikimybė gali likti toje pačioje pakopoje."
        )

    with tab_csv:
        st.write(
            "CSV turi turėti tuos pačius požymius kaip UCI rinkinys. "
            "Jei yra stulpelis `default`, skaičiuojamos ir metrikos."
        )
        uploaded = st.file_uploader("Įkelkite CSV", type=["csv"])
        if uploaded is not None:
            extra = pd.read_csv(uploaded)
            from src.data import clean_categories, _normalize_columns

            extra = clean_categories(_normalize_columns(extra, require_target=False))
            extra = extra.copy()
            schema = joblib.load(MODELS_DIR / "schema_example.joblib")
            X = extra.reindex(columns=list(schema.columns), fill_value=0)
            p_raw = model.predict_proba(X)
            p_cal = calibrator.transform(p_raw)
            extra["p_raw"] = p_raw
            extra["p_cal"] = p_cal
            extra["yhat"] = decision_from_threshold(p_cal, t)
            st.dataframe(extra.head(30), hide_index=True, width="stretch")
            if "default" in extra.columns:
                y = extra["default"].astype(int).to_numpy()
                from src.metrics import evaluate_probabilities

                m = evaluate_probabilities(y, p_cal, t, float(results["c_fn"]), float(results["c_fp"]))
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("PR-AUC", fmt(m["pr_auc"]))
                k2.metric("Brier", fmt(m["brier"]))
                k3.metric("FN", m["FN"])
                k4.metric("Cost", f"{m['cost']:.0f}")


def page_ai() -> None:
    st.header("AI naudojimo žurnalas")
    path = PROJECT_ROOT / "reports" / "AI_NAUDOJIMO_ZURNALAS.md"
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.warning("Žurnalas dar nesukurtas.")


def page_code() -> None:
    st.header("Projekto kodas")
    files = sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in SRC_DIR.glob("*.py"))
    files = ["run_experiment.py", "svetaine.py"] + files
    choice = st.selectbox("Failas", files)
    st.code(read_source(choice), language="python")


def page_repro(results: dict) -> None:
    st.header("Atkuriamumas")
    st.write(
        "Saugykla: [https://github.com/Meilaa/Kredito-sipareigojim-nevykdymo-rizikos-prognoz-]"
        "(https://github.com/Meilaa/Kredito-sipareigojim-nevykdymo-rizikos-prognoz-)"
    )
    st.write(
        f"Atsitiktinė sėkla **{RANDOM_SEED}**. Viena komanda pagrindiniam eksperimentui: "
        "`python run_experiment.py`. Svetainė: `streamlit run svetaine.py`."
    )
    st.code(
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "python run_experiment.py\n"
        "streamlit run svetaine.py\n"
        "jupyter notebook notebooks/egzamino_sprendimas.ipynb",
        language="bash",
    )
    st.subheader("Sprendimo ribos")
    st.markdown(
        """
        - Duomenys surinkti Taivane 2005 m. – rezultatų negalima perkelti į dabartinę Lietuvos rinką.
        - Modelis neduoda priežastinio paaiškinimo ir neturi būti vienintelis kredito sprendimas.
        - Demografiniai požymiai gali veikti kaip proxy; grupių skirtumai reikalauja atitikties vertinimo.
        - Slenkstis priklauso nuo pasirinktų `C_FN` ir `C_FP`; kitos kainos duotų kitą t.
        - Klasių disbalansas (~22 %) daro accuracy klaidinančia metrika.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Kredito default rizikos egzaminas",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    results = load_results()
    if results is None:
        st.error("Nerasti eksperimento rezultatai. Paleiskite `python run_experiment.py`.")
        st.stop()

    pages = {
        "Pradžia": page_home,
        "Problema ir duomenys": page_problem,
        "Formulės ir kodas": page_formulas,
        "Modeliai": page_models,
        "Rezultatai": page_results,
        "Kalibracija ir kaina": page_calibration,
        "Grupės ir klaidos": page_groups,
        "Abliacija ir atsparumas": page_ablation,
        "Gyvas bandymas": page_live,
        "AI žurnalas": lambda _r: page_ai(),
        "Kodas": lambda _r: page_code(),
        "Atkuriamumas ir ribos": page_repro,
    }
    st.sidebar.markdown("### Turinys")
    choice = st.sidebar.radio("Skyrius", list(pages.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption("Meila Andriuškevičiūtė · DISfm-26")
    st.sidebar.caption(f"Sėkla {RANDOM_SEED} · C_FN={results['c_fn']} · C_FP={results['c_fp']}")
    pages[choice](results)


if __name__ == "__main__":
    main()
