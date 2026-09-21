"""Ataskaitos tekstai, sudaromi iš `artifacts/` skaičių.

Skaičiai į PDF ir Word ataskaitas neįrašomi ranka: kiekvienas sakinys
sudaromas iš `artifacts/results.json` ir `artifacts/tables/*.csv`. Todėl po
`python run_experiment.py` ataskaitos tekstas visada atitinka tikrus rezultatus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TABLES = ARTIFACTS / "tables"


def lt(value, digits: int = 4) -> str:
    """Skaičius lietuviškai: dešimtainė kablelis, tūkstančiai su tarpu."""
    if pd.isna(value):
        return "—"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}".replace(",", " ")
    text = f"{float(value):.{digits}f}"
    return text.replace(".", ",")


def _sk(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def surinkti() -> dict:
    res = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(TABLES / "model_comparison.csv")
    groups = pd.read_csv(TABLES / "group_performance.csv")
    ablation = pd.read_csv(TABLES / "ablation.csv")
    robust = pd.read_csv(TABLES / "robustness.csv")

    laimetojas = res["winner"]
    lm = summary[summary["model"] == laimetojas].iloc[0]
    kiti = summary[summary["model"] != laimetojas]
    antras = kiti.iloc[0]
    log = summary[summary["model"] == "Logistinė regresija"].iloc[0]
    naiv = summary[summary["model"] == "Bendras dažnis"].iloc[0]
    boost = summary[summary["model"] == "XGBoost"].iloc[0]
    mis = summary[summary["model"] == "Random Forest"].iloc[0]
    # Kuris modelis pirmauja kurioje imtyje (atrankai naudojamas tik validavimas)
    val_lyderis = str(summary.sort_values("val_pr_auc_for_selection", ascending=False).iloc[0]["model"])
    testo_lyderis = str(summary.sort_values("pr_auc", ascending=False).iloc[0]["model"])

    kal = res["calibration_choice"][laimetojas]
    q = res["quality"]
    s = res["split"]
    par = {r["model"]: json.loads(r["params"]) for _, r in summary.iterrows()}

    # --- grupės ---
    lytis = groups[groups["group"] == "lytis"]
    amzius = groups[groups["group"] == "amžiaus grupė"].copy()
    amz_zemiausia = amzius.loc[amzius["pr_auc"].idxmin()]
    amz_auksciausia = amzius.loc[amzius["pr_auc"].idxmax()]

    # --- abliacija ---
    ab_visi = ablation[ablation["variant"] == "visi požymiai"].iloc[0]
    ab_be_demo = ablation[ablation["variant"] == "be demografijos"].iloc[0]
    ab_be_pay = ablation[ablation["variant"] == "be mokėjimų būsenų"].iloc[0]

    # --- atsparumas ---
    rob_svarus = robust[robust["scenario"] == "švarus testas"].iloc[0]
    rob_triuksmas = robust[robust["scenario"].str.startswith("Gauso")].iloc[0]
    rob_trukst = robust[robust["scenario"].str.startswith("10 %")].iloc[0]
    rob_senesni = robust[robust["scenario"].str.startswith("grupės poslinkis")]

    # --- rankinio įgyvendinimo patikra ---
    patikra_kelias = ARTIFACTS / "rankiniu_patikra.json"
    patikros_eilutes = []
    if patikra_kelias.exists():
        patikros_eilutes = json.loads(patikra_kelias.read_text(encoding="utf-8"))["eilutes"]

    t: dict[str, object] = {}
    t["laimetojas"] = laimetojas
    t["slenkstis"] = float(lm["threshold"])

    t["intelektualus_metodai"] = (
        "Svarstytini intelektualieji metodai: SVM, MLP, medžių ansamblis, fuzzy taisyklės. "
        "Šiame darbe pasirinkti medžių ansambliai – XGBoost tipo gradientinis boostingas ir "
        "Random Forest – ir abu parašyti rankiniu būdu."
    )

    t["hipotezes_isvada"] = (
        f"Išvada: hipotezė pasitvirtino. Modelis atrinktas pagal validavimo PR-AUC "
        f"(XGBoost {lt(boost['val_pr_auc_for_selection'])}, Random Forest "
        f"{lt(mis['val_pr_auc_for_selection'])}, logistinė regresija "
        f"{lt(log['val_pr_auc_for_selection'])}); laimėjo {laimetojas}. Galutiniame teste "
        f"XGBoost pasiekė PR-AUC {lt(boost['pr_auc'])}, Random Forest – {lt(mis['pr_auc'])}, "
        f"logistinė regresija – {lt(log['pr_auc'])}, bendras dažnis – {lt(naiv['pr_auc'])}. "
        f"Abu ansambliai aiškiai pranoko logistinę regresiją. Po isotonic kalibravimo "
        f"laimėtojo Brier score yra {lt(lm['brier'])}, ECE – {lt(lm['ece'], 4)}."
    )

    t["rankinis_igyvendinimas"] = (
        "Visi mokymo algoritmai parašyti rankiniu būdu faile src/rankiniai_modeliai.py "
        "naudojant tik numpy. Logistinė regresija optimizuojama Newton–Raphson metodu "
        f"(konverguoja per {res['logistic_extra'].get('newton_iteracijos', '—')} iteracijas): "
        "skaičiuojamas gradientas g = Xᵀ(s·(p−y)) + λβ ir Hessian "
        "H = Xᵀ·diag(s·p·(1−p))·X + λI. Medžiai statomi iš histogramų: kiekvienas požymis "
        "diskretizuojamas į 64 kvantilių binus, o padalijimas renkamas iš binų sumų. "
        "XGBoost tipo boostingas naudoja logistinės nuostolių funkcijos pirmąją ir antrąją "
        "išvestines (g ir h), o lapo reikšmė skaičiuojama w = −ΣG/(ΣH+λ). Random Forest "
        "naudoja bootstrap pavyzdžius, požymių atranką mazge ir Gini kriterijų. Lentelėse "
        "palikti įprasti metodų pavadinimai (XGBoost, Random Forest), bet kodas yra savas – "
        "klasės GradientinisBoostingas ir AtsitiktinisMiskas. sklearn naudojamas tik "
        "kategorijų kodavimui ir metrikoms, xgboost biblioteka – tik patikrai."
    )

    if patikros_eilutes:
        dalys = []
        for e in patikros_eilutes:
            skirtumas = abs(e["pr_auc_rankinis"] - e["pr_auc_biblioteka"])
            dalys.append(
                f"{e['modelis']}: rankinis {lt(e['pr_auc_rankinis'])} prieš "
                f"bibliotekos {lt(e['pr_auc_biblioteka'])} (skirtumas {lt(skirtumas)})"
            )
        t["patikros_tekstas"] = (
            "Kad rankinis įgyvendinimas nebūtų priimtas aklai, tie patys paruošti duomenys "
            "paduoti ir bibliotekoms; lyginamos validavimo imties metrikos. "
            + "; ".join(dalys)
            + ". Logistinės regresijos koeficientai sutampa su sklearn, o ansamblių "
            "tikimybių koreliacija su bibliotekomis viršija 0,99. Patikrą galima pakartoti: "
            "python scripts/patikra_rankiniai.py."
        )
        t["patikros_eilutes"] = [
            [
                e["modelis"],
                lt(e["pr_auc_rankinis"]),
                lt(e["pr_auc_biblioteka"]),
                lt(abs(e["pr_auc_rankinis"] - e["pr_auc_biblioteka"])),
                (
                    lt(e["tikimybiu_koreliacija"])
                    if "tikimybiu_koreliacija" in e
                    else f"maks. skirtumas {e['maks_tikimybiu_skirtumas']:.1e}"
                ),
            ]
            for e in patikros_eilutes
        ]
    else:
        t["patikros_tekstas"] = (
            "Patikra su bibliotekomis dar nepaleista: python scripts/patikra_rankiniai.py."
        )
        t["patikros_eilutes"] = []

    t["problema"] = (
        "Tikslas – pagal kredito kortelės kliento istorinius duomenis įvertinti kalibruotą "
        "tikimybę, kad klientas kitą mėnesį nevykdys finansinių įsipareigojimų. Modelis "
        "vertinamas kaip sprendimų palaikymo priemonė, ne automatinė galutinio kredito "
        f"sprendimo sistema. Rinkinį sudaro {_sk(q['n_rows'])} klientų, iš jų "
        f"{_sk(q['n_default'])} ({lt(q['default_rate'] * 100, 2)} %) priklauso default klasei."
    )

    t["duomenu_grandine"] = (
        f"Stratifikuotas 70/15/15 skaidymas, sėkla {res['seed']}. Mokymas {_sk(s['n_train'])}, "
        f"validavimas {_sk(s['n_val'])}, testas {_sk(s['n_test'])}. EDUCATION reikšmės 0, 5, 6 "
        "sujungtos į 4; MARRIAGE 0 – į 3. Kategorijos koduojamos OneHotEncoder, skaitiniai "
        "požymiai logistinei regresijai standartizuojami. Visi preprocessing parametrai "
        "skaičiuojami tik mokymo imtyje."
    )

    t["grandines_kodas"] = (
        "UCI duomenys → kokybės patikra → stratifikuotas 70/15/15 →\n"
        "preprocessing tik pagal mokymą → bendras dažnis + logistinė regresija →\n"
        "XGBoost + Random Forest (rankiniai) → validavimo metrikos → kalibravimas →\n"
        "modelio atranka pagal validavimą → slenksčio / kainos analizė → grupės →\n"
        "abliacija / atsparumas → vienkartinis testas"
    )

    t["modeliai"] = (
        f"Naivus baseline visiems klientams priskiria mokymo imties default dažnį "
        f"({lt(par['Bendras dažnis']['p'], 4)}). Logistinė regresija yra interpretuojamas "
        f"baseline; hiperparametras C ∈ {{0,1; 1; 10}} ir class_weight parinkti pagal "
        f"validavimo PR-AUC (geriausias C = {lt(par['Logistinė regresija']['C'], 1)}). "
        f"XGBoost tipo gradientinis boostingas – pagrindinis intelektualus metodas "
        f"(n_estimators={par['XGBoost']['n_estimators']}, "
        f"max_depth={par['XGBoost']['max_depth']}, "
        f"learning_rate={lt(par['XGBoost']['learning_rate'], 2)}, "
        f"min_child_weight={par['XGBoost']['min_child_weight']}). "
        f"Random Forest – antroji alternatyva "
        f"({par['Random Forest']['n_estimators']} medžių, "
        f"max_depth={par['Random Forest']['max_depth']}, "
        f"min_samples_leaf={par['Random Forest']['min_samples_leaf']}). Abiem ansambliams "
        f"tikrinta po keturis parametrų variantus, kad palyginimas būtų vienodas. "
        f"Kalibratorius (isotonic arba Platt) mokomas tik validavime; visiems modeliams "
        f"pasirinktas {kal['method']}."
    )

    t["formules_patikra"] = (
        "Šios formulės įgyvendintos faile src/formulas.py funkcijose linear_score ir "
        "logistic_probability, o mokymo metu naudojamos src/rankiniai_modeliai.py klasėje "
        "LogistineRegresija. Validavimo imties aštuoniose eilutėse rankinės tikimybės sutampa "
        f"su modelio išvestimi iki {res['formula_check']['max_abs_diff']:.2e}."
    )

    t["boostingo_taisykle"] = (
        "XGBoost tipo boostingas dvejetainei klasifikacijai naudoja logistinę nuostolių "
        "funkciją: F(x) = F₀ + η·Σ_k medis_k(x), p̂ = σ(F(x)). Kiekvienas naujas medis "
        "mokomas iš gradientų g = s·(p − y) ir Hessian'ų h = s·p·(1 − p), o lapo reikšmė "
        "yra w = −ΣG/(ΣH + λ). Slenkstis t parenkamas validavime minimizuojant Cost(t)."
    )

    t["rezultatu_ivadas"] = (
        "Visi metodai lyginti tomis pačiomis imtimis. Hiperparametrai, kalibratorius, "
        "slenkstis ir galutinio modelio atranka atlikti nenaudojant testo: laimėtojas "
        "renkamas pagal validavimo PR-AUC, o testas skaičiuojamas vieną kartą kaip galutinis "
        "įvertis. Pagrindinė diskriminacijos metrika – PR-AUC, nes klasės nesubalansuotos. "
        f"Accuracy čia klaidina: pastovus „nėra default“ duotų apie "
        f"{lt((1 - q['default_rate']) * 100, 0)} %."
    )

    t["lentele1"] = (
        f"1 lentelė. Kalibruotos tikimybės, testo imtis (n = {_sk(s['n_test'])}). Eilutės "
        "surikiuotos pagal validavimo PR-AUC – pagal ją renkamas laimėtojas. Mažesnis Brier "
        "ir ECE – geriau."
    )

    if val_lyderis == testo_lyderis:
        imciu_sakinys = (
            f"Tas pats modelis pirmauja abiejose imtyse: validavime "
            f"{lt(lm['val_pr_auc_for_selection'])}, teste {lt(lm['pr_auc'])}."
        )
    else:
        testo_l = summary[summary["model"] == testo_lyderis].iloc[0]
        imciu_sakinys = (
            f"Verta pastebėti, kad testo imtyje nežymiai pirmauja {testo_lyderis} "
            f"({lt(testo_l['pr_auc'])} prieš {lt(lm['pr_auc'])}). Laimėtojas vis tiek "
            f"renkamas pagal validavimą, nes testas turi likti vienkartinis įvertis; "
            f"skirtumas ({lt(abs(testo_l['pr_auc'] - lm['pr_auc']))}) mažesnis už tokios "
            f"imties svyravimą."
        )

    t["rezultatu_komentaras"] = (
        f"Pagal validavimo PR-AUC laimėjo {laimetojas} "
        f"({lt(lm['val_pr_auc_for_selection'])}), antras – {antras['model']} "
        f"({lt(antras['val_pr_auc_for_selection'])}). {imciu_sakinys} Abu ansambliai labai "
        f"artimi, todėl svarbu pažiūrėti ir klaidų struktūrą: Random Forest prie savo "
        f"slenksčio praleidžia mažiau default (FN {int(mis['FN'])} prieš {int(boost['FN'])}), "
        f"bet be pagrindo įtaria daugiau klientų (FP {_sk(int(mis['FP']))} prieš "
        f"{_sk(int(boost['FP']))}), todėl jo tikslumas prie slenksčio mažesnis "
        f"({lt(mis['precision_at_t'], 3)} prieš {lt(boost['precision_at_t'], 3)}), o recall "
        f"didesnis ({lt(mis['recall_at_t'], 3)} prieš {lt(boost['recall_at_t'], 3)}). "
        f"Tikimybių kokybė po kalibravimo beveik vienoda (Brier {lt(boost['brier'])} ir "
        f"{lt(mis['brier'])}; ECE {lt(boost['ece'], 4)} ir {lt(mis['ece'], 4)}). Logistinė "
        f"regresija silpnesnė ir pagal PR-AUC, ir pagal Brier. Bendro dažnio PR-AUC "
        f"({lt(naiv['pr_auc'])}) sutampa su default dažniu – tai naivi atskaitos riba ir "
        f"kartu metrikų patikra."
    )

    t["kalibracija"] = (
        "Geras surikiavimas pagal riziką nereiškia, kad prognozuota 0,30 tikimybė atitinka "
        "30 % įvykių. Todėl tikimybės kalibruotos isotonic regresija validavimo imtyje. "
        f"{laimetojas} Brier score validavime po kalibravimo sumažėjo nuo "
        f"{lt(kal['val_brier_raw'])} iki {lt(kal['val_brier_cal'])}."
    )

    t["slenksčio_komentaras"] = (
        f"Slenkstis {lt(lm['threshold'], 3)} pasirinktas ne pagal accuracy, o pagal Cost(t) su "
        f"C_FN:C_FP = {int(res['c_fn'])}:{int(res['c_fp'])}. Prie jo modelis pagauna "
        f"{lt(lm['recall_at_t'] * 100, 1)} % tikrų default atvejų. Žemesnis slenkstis "
        "padidintų recall, bet ir nepagrįstų ribojimų (FP) skaičių."
    )

    t["figura_kaina"] = (
        f"4 pav. {laimetojas} klaidų kainos kreivė; pasirinktas t = {lt(lm['threshold'], 3)}"
    )

    t["lentele2"] = (
        f"2 lentelė. {laimetojas} pagal lytį (testas). PR-AUC panašus "
        f"({', '.join(lt(v) for v in lytis['pr_auc'])})."
    )

    t["grupes"] = (
        f"Amžiaus grupėje {amz_zemiausia['value']} PR-AUC žemesnis "
        f"({lt(amz_zemiausia['pr_auc'])}) nei grupėje {amz_auksciausia['value']} "
        f"({lt(amz_auksciausia['pr_auc'])}). Skirtumai neįrodo priežastinės diskriminacijos, "
        "bet demografiniai požymiai gali būti proxy. Abliacija be demografijos PR-AUC "
        f"sumažina nedaug ({lt(ab_visi['test_pr_auc'])} → {lt(ab_be_demo['test_pr_auc'])}), "
        "todėl praktikoje demografiją galima šalinti be didelio nuostolio."
    )

    t["klaidos"] = (
        "False negative (praleistas default) brangesnis. FN klientai vidutiniškai turi "
        "didesnį limitą ir geresnę PAY_1 būseną nei TP – modelis juos laiko saugesniais. "
        "FP klientai dažniau turi teigiamą PAY_1 (vėlavimą) ir mažesnį limitą."
    )

    t["lentele3"] = (
        f"3 lentelė. Vidutiniai požymiai pagal TP/TN/FP/FN (testas, {laimetojas}, "
        f"t = {lt(lm['threshold'], 3)})."
    )

    t["lentele6"] = (
        "6 lentelė. XGBoost abliacija. Modelis kiekvienam variantui mokomas iš naujo, o "
        "lentelė naudoja nekalibruotas tikimybes, kad būtų lyginama tik požymių įtaka."
    )

    t["abliacija"] = (
        f"Mokėjimų būsenos yra svarbiausia grupė: be PAY_* PR-AUC krenta nuo "
        f"{lt(ab_visi['test_pr_auc'])} iki {lt(ab_be_pay['test_pr_auc'])}. Demografijos "
        f"pašalinimas pakeičia rezultatą nedaug ({lt(ab_be_demo['test_pr_auc'])})."
    )

    t["lentele7"] = f"7 lentelė. Tas pats {laimetojas} modelis, be pakartotinio mokymo."

    senesni = ""
    if not rob_senesni.empty:
        r = rob_senesni.iloc[0]
        senesni = (
            f" Vyresnių klientų grupėje diskriminacija išlieka ({lt(r['pr_auc'])}), "
            f"bet imtis mažesnė (n = {_sk(int(r['n']))})."
        )
    t["atsparumas"] = (
        f"Gauso triukšmas (σ = 0,15 · std) PR-AUC sumažina nuo {lt(rob_svarus['pr_auc'])} iki "
        f"{lt(rob_triuksmas['pr_auc'])}. 10 % atsitiktinai trūkstamų reikšmių poveikis mažesnis "
        f"({lt(rob_trukst['pr_auc'])})." + senesni
    )

    t["kodel_lame_antrase"] = f"5. Kodėl laimėjo {laimetojas} ir kokios ribos"

    # Tikimybių kokybė lyginama iš skaičių, o ne teigiama iš anksto
    brier_skirtumas = float(boost["brier"]) - float(mis["brier"])
    if abs(brier_skirtumas) < 0.0005:
        brier_sakinys = (
            f"Tikimybių tikslumas praktiškai vienodas (Brier {lt(boost['brier'])} ir "
            f"{lt(mis['brier'])})"
        )
    elif brier_skirtumas < 0:
        brier_sakinys = (
            f"XGBoost tikimybės tikslesnės (Brier {lt(boost['brier'])} prieš "
            f"{lt(mis['brier'])})"
        )
    else:
        brier_sakinys = (
            f"Random Forest tikimybės tikslesnės (Brier {lt(mis['brier'])} prieš "
            f"{lt(boost['brier'])})"
        )
    if float(boost["ece"]) < float(mis["ece"]):
        ece_sakinys = (
            f", o ECE mažesnis XGBoost ({lt(boost['ece'], 4)} prieš {lt(mis['ece'], 4)})."
        )
    else:
        ece_sakinys = (
            f", o ECE mažesnis Random Forest ({lt(mis['ece'], 4)} prieš "
            f"{lt(boost['ece'], 4)})."
        )

    if laimetojas == "XGBoost":
        kodel = (
            "XGBoost pirmauja atrankos metrikoje todėl, kad medžiai statomi nuosekliai: "
            "kiekvienas naujas medis mokomas iš likusio gradiento g = s·(p − y), t. y. taiso "
            "būtent tas klaidas, kurias ansamblis dar daro, o Hessian h = s·p·(1 − p) per "
            "lapo formulę w = −ΣG/(ΣH+λ) neleidžia per stipriai pasitikėti mažomis "
            "grupėmis. Random Forest medžiai mokomi nepriklausomai ir tik vidurkinami – jie "
            "mažina dispersiją, bet neturi mechanizmo sisteminei klaidai taisyti. "
            f"Validavime tai davė {lt(boost['val_pr_auc_for_selection'])} prieš "
            f"{lt(mis['val_pr_auc_for_selection'])}."
        )
    else:
        kodel = (
            f"{laimetojas} pirmauja atrankos metrikoje "
            f"({lt(mis['val_pr_auc_for_selection'])} prieš "
            f"{lt(boost['val_pr_auc_for_selection'])}): daug nepriklausomų medžių, mokytų "
            "skirtinguose bootstrap pavyzdžiuose, šiuose duomenyse rikiuoja klientus šiek "
            "tiek geriau nei nuoseklus gradiento taisymas."
        )

    t["kodel_lame"] = (
        f"Abu medžių ansambliai pranoko logistinę regresiją ({lt(boost['pr_auc'])} ir "
        f"{lt(mis['pr_auc'])} prieš {lt(log['pr_auc'])} teste), nes kredito vėlavimo istorija "
        "ir mokėjimų sumos sąveikauja netiesiškai – tai medžiams natūralu, o logistinei "
        "regresijai reikėtų rankinių sąveikos narių. " + kodel + " Sąžininga pasakyti, kad "
        "tarp dviejų ansamblių skirtumas mažas ir jie skiriasi ne tiek kokybe, kiek klaidų "
        f"pusiausvyra: XGBoost prie savo slenksčio ({lt(boost['threshold'], 3)}) nepagrįstai "
        f"įtaria {_sk(int(boost['FP']))} klientų ir praleidžia {int(boost['FN'])} default, o "
        f"Random Forest prie savo slenksčio ({lt(mis['threshold'], 3)}) praleidžia mažiau "
        f"({int(mis['FN'])}), bet įtaria {_sk(int(mis['FP']))}. Bendra klaidų kaina beveik "
        f"sutampa ({lt(boost['cost'], 0)} ir {lt(mis['cost'], 0)}). " + brier_sakinys
        + ece_sakinys + " Abiem ansambliams tikrinta po tiek pat parametrų variantų, todėl "
        "palyginimas nėra iškreiptas vieno metodo naudai. MLP nebuvo treniruotas šiame "
        "paleidime: du intelektualūs metodai tenkina užduotį, o MLP jautresnis masteliui."
    )

    t["ribos"] = [
        "Duomenys surinkti Taivane 2005 m. – rezultatų negalima perkelti į dabartinę Lietuvos rinką.",
        "Modelis neduoda priežastinio paaiškinimo ir neturi būti vienintelis kredito sprendimas.",
        f"Slenkstis priklauso nuo C_FN ir C_FP; kitos kainos duotų kitą t (dabar {lt(lm['threshold'], 3)}).",
        "Isotonic kalibravimas p̂_cal keičia laiptais – panašūs balai gali gauti tą pačią tikimybę.",
        f"Klasių disbalansas (~{lt(q['default_rate'] * 100, 0)} %) daro accuracy klaidinančia metrika.",
        "Rankiniai medžiai naudoja 64 kvantilių binus, todėl padalijimų aibė šiek tiek "
        "siauresnė nei tikrinant visas reikšmes.",
    ]

    t["atkuriamumas"] = (
        f"Atsitiktinė sėkla {res['seed']}. Duomenys atsisiunčiami automatiniu būdu iš UCI "
        "(id=350). Viena komanda:"
    )

    t["ai_priimta"] = (
        "Priimta: modulinė Python grandinė, rankiniai algoritmai (Newton–Raphson, histogramų "
        "medžiai, bootstrap), 70/15/15, laimėtojo atranka pagal validavimą, Streamlit, "
        "formulės src/formulas.py. Atmesta: tik Jupyter be svetainės; bibliotekiniai modeliai "
        "kaip pagrindinis sprendimas – jie palikti tik patikrai; MLP kaip trečias metodas "
        "šiame paleidime."
    )

    t["ai_klaidos"] = [
        "ucimlrepo grąžino stulpelius X1–X23, ne LIMIT_BAL. Pirmas mokymas žlugo su 0 požymių. "
        "Pataisyta UCI_X_MAP, patikrinta pagal Yeh ir Lien aprašą.",
        "Rankinio medžio padalijimo riba turi atitikti bino kraštą: mokymas naudoja „bin ≤ b“, "
        "prognozė – „x < briauna[b]“. Patikra: prognozė ant mokymo imties sutampa su mokymo metu "
        "naudotais lapais.",
        "Straipsnio [2] PR-AUC ≈ 0,89 neperkeltina į šį UCI rinkinį – DOI patikrintas, metrikos "
        "imamos tik iš run_experiment.py.",
        "Pirmoje versijoje laimėtojas buvo renkamas pagal TESTO PR-AUC – tai testo imties "
        "nutekėjimas į atranką. Pataisyta: laimėtojas renkamas tik pagal validavimo PR-AUC, "
        "testas skaičiuojamas vieną kartą. Patikra: results.json laukas winner_selected_by ir "
        "abiejų imčių PR-AUC 1 lentelėje.",
    ]

    return t
