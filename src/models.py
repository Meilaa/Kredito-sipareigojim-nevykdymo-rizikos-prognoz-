"""Modelių mokymas rankiniu būdu parašytais algoritmais.

Baseline: bendras default dažnis ir logistinė regresija.
Intelektualūs metodai: XGBoost tipo gradientinis boostingas ir Random Forest.
Pavadinimai lentelėse palikti angliški, nes tai standartiniai metodų vardai, bet
patys algoritmai parašyti rankiniu būdu, o ne paimti iš bibliotekų.

Visi mokymo algoritmai yra faile `src/rankiniai_modeliai.py` – parašyti tik su
numpy, be jokios mokymo bibliotekos. `sklearn` čia naudojamas tik kategorijų
kodavimui (`build_preprocessor`) ir PR-AUC apskaičiavimui.

Hiperparametrai visiems modeliams parenkami TIK validavimo imtyje.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import RANDOM_SEED
from src.formulas import logistic_probability, linear_score
from src.preprocess import build_preprocessor
from src.rankiniai_modeliai import (
    AtsitiktinisMiskas,
    GradientinisBoostingas,
    Grandine,
    LogistineRegresija,
)


@dataclass
class TrainedModel:
    name: str
    kind: str
    pipeline: Any
    best_params: dict
    val_pr_auc: float
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Grąžina default tikimybę (teigiamos klasės stulpelį)."""
        if self.kind == "constant":
            return np.full(len(X), float(self.extra["p"]), dtype=float)
        proba = self.pipeline.predict_proba(X)
        return proba[:, 1]


def _pr_auc(y_true, p_hat) -> float:
    """PR-AUC – pagrindinė atrankos metrika nesubalansuotoms klasėms."""
    return float(average_precision_score(y_true, p_hat))


# ============================================================
# 1. Naivus baseline: bendras default dažnis
# ============================================================


class BendrasDaznis:
    """Visiems klientams priskiria tą pačią tikimybę – mokymo imties dažnį.

        p = (1/N) · Σ y_i
    """

    def __init__(self) -> None:
        self.p = 0.0

    def fit(self, y_train: pd.Series) -> "BendrasDaznis":
        self.p = float(np.mean(y_train))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p, dtype=float)


def train_constant(y_train: pd.Series) -> TrainedModel:
    modelis = BendrasDaznis().fit(y_train)
    return TrainedModel(
        name="Bendras dažnis",
        kind="constant",
        pipeline=modelis,
        best_params={"p": modelis.p},
        val_pr_auc=float("nan"),
        notes="Naivus baseline be požymių. Jo PR-AUC lygus default dažniui.",
        extra={"p": modelis.p},
    )


# ============================================================
# 2. Logistinė regresija (rankinis Newton–Raphson)
# ============================================================


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> TrainedModel:

    # ============================================================
    # 1. Parametrai, kuriuos norime išbandyti
    # ============================================================
    #
    # C valdo L2 baudos stiprumą: λ = 1/C.
    # Mažas C -> stipri bauda -> koeficientai traukiami į nulį.

    parametru_variantai = [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ]

    # ============================================================
    # 2. Čia saugosime geriausią variantą
    # ============================================================

    geriausia_grandine = None
    geriausias_pr_auc = -1.0
    geriausi_parametrai = None

    # ============================================================
    # 3. Išbandome kiekvieną variantą
    # ============================================================

    for params in parametru_variantai:

        # Logistinei regresijai skaitiniai požymiai standartizuojami,
        # nes Newton metodas jautrus labai skirtingiems masteliams
        # (LIMIT_BAL yra šimtai tūkstančių, AGE – dešimtys).
        preprocessor = build_preprocessor(X_train, scale_numeric=True)

        # --------------------------------------------------------
        # 4. Sukuriame savo logistinės regresijos modelį
        # --------------------------------------------------------
        #
        # z_i = β₀ + β₁x_i1 + ... + β_p x_ip
        # p_i = 1 / (1 + e^{-z_i})
        #
        # Minimizuojama:
        #   L(β) = Σ s_i·[-y_i·log(p_i) - (1-y_i)·log(1-p_i)]
        #          + (1/(2C))·Σ β_j²

        modelis = LogistineRegresija(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=100,
            tol=1e-10,
        )

        grandine = Grandine(preprocessor, modelis)

        # ========================================================
        # 5. ČIA REALIAI VYKSTA MOKYMAS
        # ========================================================

        grandine.fit(X_train, y_train)

        # Kiekvienoje iteracijoje viduje (src/rankiniai_modeliai.py):
        #
        #   1. p = sigmoide(Xβ)
        #   2. gradientas   g = Xᵀ(s·(p - y)) + λβ
        #   3. Hessian      H = Xᵀ·diag(s·p·(1-p))·X + λI
        #   4. β ← β - H⁻¹g
        #
        # Sustoja, kai žingsnis mažesnis už tol (~8 iteracijos).

        # ========================================================
        # 6. Vertiname VALIDAVIMO imtyje
        # ========================================================

        p_val = grandine.predict_proba(X_val)[:, 1]
        pr_auc = _pr_auc(y_val, p_val)

        # ========================================================
        # 7. Jei geresnis – išsaugome
        # ========================================================

        if pr_auc > geriausias_pr_auc:
            geriausias_pr_auc = pr_auc
            geriausia_grandine = grandine
            geriausi_parametrai = params

    # ============================================================
    # 8. Grąžiname geriausią variantą kartu su koeficientais
    # ============================================================

    clf = geriausia_grandine.named_steps["clf"]
    return TrainedModel(
        name="Logistinė regresija",
        kind="logistic",
        pipeline=geriausia_grandine,
        best_params=geriausi_parametrai,
        val_pr_auc=geriausias_pr_auc,
        notes=(
            "Interpretuojamas baseline. Tikimybė skaičiuojama formule "
            "p̂ = 1/(1+e^{-z}), z = β₀ + Xβ. Koeficientai gauti Newton–Raphson "
            "metodu, parašytu src/rankiniai_modeliai.py."
        ),
        extra={
            "intercept": float(clf.intercept_),
            "coefficients": clf.coef_.tolist(),
            "feature_names": list(
                geriausia_grandine.named_steps["prep"].get_feature_names_out()
            ),
            "newton_iteracijos": int(clf.n_iter_),
        },
    )


# ============================================================
# 3. Random Forest (atsitiktinis miškas, rankinis)
# ============================================================


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> TrainedModel:

    # ============================================================
    # 1. Parametrai, kuriuos norime išbandyti
    # ============================================================

    # Tiek variantų, kiek ir boostingui – kad palyginimas būtų sąžiningas.
    parametru_variantai = [
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 20},
    ]

    geriausia_grandine = None
    geriausias_pr_auc = -1.0
    geriausi_parametrai = None

    # ============================================================
    # 2. Išbandome kiekvieną variantą
    # ============================================================

    for params in parametru_variantai:

        # Medžiams standartizuoti nereikia: padalijimas „x < riba"
        # nesikeičia, jei požymį padauginam ar pastumiam.
        preprocessor = build_preprocessor(X_train, scale_numeric=False)

        # --------------------------------------------------------
        # 3. Sukuriame savo atsitiktinį mišką
        # --------------------------------------------------------
        #
        # Kiekvienas medis mokomas savo bootstrap pavyzdyje, o
        # kiekviename mazge tikrinama tik √d požymių.
        #
        # Galutinė tikimybė – medžių vidurkis:
        #
        #   p(x) = (1/K) · Σ_k medis_k(x)

        modelis = AtsitiktinisMiskas(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            class_weight="balanced",
            n_binu=64,
            random_state=RANDOM_SEED,
        )

        grandine = Grandine(preprocessor, modelis)

        # ========================================================
        # 4. ČIA REALIAI VYKSTA MIŠKO MOKYMAS
        # ========================================================

        grandine.fit(X_train, y_train)

        # Viduje kiekvienam medžiui:
        #
        #   1. bootstrap: N klientų su pakartojimu
        #   2. mazge tikrinami tik √d požymiai
        #   3. padalijimas renkamas pagal Gini:
        #
        #        Gini = 1 - p₁² - p₀²
        #
        #      ieškoma mažiausio svertinio vaikų Gini
        #   4. lapo prognozė – svertinė default klientų dalis

        # ========================================================
        # 5. Vertiname validavimo imtyje
        # ========================================================

        p_val = grandine.predict_proba(X_val)[:, 1]
        pr_auc = _pr_auc(y_val, p_val)

        if pr_auc > geriausias_pr_auc:
            geriausias_pr_auc = pr_auc
            geriausia_grandine = grandine
            geriausi_parametrai = params

    return TrainedModel(
        name="Random Forest",
        kind="random_forest",
        pipeline=geriausia_grandine,
        best_params=geriausi_parametrai,
        val_pr_auc=geriausias_pr_auc,
        notes=(
            "Random Forest tipo medžių ansamblis, parašytas rankiniu būdu "
            "(src/rankiniai_modeliai.py, klasė AtsitiktinisMiskas): bootstrap, požymių "
            "atranka mazge, Gini kriterijus. Tikimybė – medžių lapų vidurkis."
        ),
        extra={"max_features": "sqrt", "class_weight": "balanced"},
    )


# ============================================================
# 4. XGBoost tipo gradientinis boostingas (rankinis)
# ============================================================


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> TrainedModel:

    # ============================================================
    # 1. Apskaičiuojame klasių balansą
    # ============================================================

    # Kiek mokymo duomenyse yra klientų su y = 1 (default)
    n_pos = float(np.sum(y_train == 1))

    # Kiek klientų yra su y = 0 (non-default)
    n_neg = float(np.sum(y_train == 0))

    # Kad retesnė klasė y=1 nebūtų ignoruojama:
    #
    #   scale_pos_weight = N_negative / N_positive

    scale_pos_weight = n_neg / max(n_pos, 1.0)

    # ============================================================
    # 2. Parametrai, kuriuos norime išbandyti
    # ============================================================

    # Mažesnis learning_rate su daugiau medžių validavime pasirodė geriau:
    # kiekvienas medis koreguoja mažiau, todėl modelis rečiau persimoko.
    parametru_variantai = [
        {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "n_binu": 64,
        },
        {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "n_binu": 64,
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "n_binu": 64,
        },
        {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 20,
            "n_binu": 128,
        },
    ]

    # ============================================================
    # 3. Čia saugosime geriausią modelį
    # ============================================================

    geriausia_grandine = None
    geriausias_pr_auc = -1.0
    geriausi_parametrai = None

    # ============================================================
    # 4. Išbandome kiekvieną parametrų variantą
    # ============================================================

    for params in parametru_variantai:

        # Duomenų paruošimas (medžiams be standartizavimo)
        preprocessor = build_preprocessor(X_train, scale_numeric=False)

        # --------------------------------------------------------
        # 5. Sukuriame savo gradientinio boostingo modelį
        # --------------------------------------------------------

        modelis = GradientinisBoostingas(

            # Kiek boosting medžių sukurti
            n_estimators=params["n_estimators"],

            # Maksimalus vieno medžio gylis
            max_depth=params["max_depth"],

            # η – kiek stipriai kiekvienas naujas medis
            # koreguoja esamą modelį
            learning_rate=params["learning_rate"],

            # Mažiausia Hessian'ų suma lape (apsauga nuo triukšmo)
            min_child_weight=params["min_child_weight"],

            # L2 bauda lapo reikšmei
            reg_lambda=1.0,

            # Kiekvienas medis mato 90 % klientų ir 90 % požymių
            subsample=0.9,
            colsample_bytree=0.9,

            # Retesnės klasės svoris
            scale_pos_weight=scale_pos_weight,

            # Į kiek kvantilių binų diskretizuojamas kiekvienas požymis
            n_binu=params["n_binu"],

            random_state=RANDOM_SEED,
        )

        # ========================================================
        # 6. Grandinė: kodavimas + modelis
        # ========================================================

        grandine = Grandine(preprocessor, modelis)

        # ========================================================
        # 7. ČIA REALIAI VYKSTA BOOSTINGO MOKYMAS
        # ========================================================

        grandine.fit(X_train, y_train)

        # Mokymo ciklas (src/rankiniai_modeliai.py) kartoja:
        #
        # 1. Apskaičiuoja dabartinę tikimybę:
        #
        #       p = sigmoide(F(x))
        #
        # 2. Kiekvienam klientui apskaičiuoja gradientą:
        #
        #       g = s · (p - y)
        #
        # 3. Apskaičiuoja Hessian:
        #
        #       h = s · p · (1 - p)
        #
        # 4. Pagal g ir h ieško geriausių medžio padalijimų:
        #
        #       gain = ½·[ G_L²/(H_L+λ) + G_R²/(H_R+λ)
        #                  - G²/(H+λ) ] - gamma
        #
        # 5. Lapo reikšmė apskaičiuojama:
        #
        #                 sum(g)
        #       w = - ----------------
        #              sum(h) + lambda
        #
        # 6. Naujas medis pridedamas prie esamo modelio:
        #
        #       F_naujas(x) = F_senas(x) + η · medis(x)
        #
        # 7. Procesas kartojamas n_estimators kartų.
        #
        # Galutinė tikimybė:
        #
        #       p = 1 / (1 + e^{-F(x)})

        # ========================================================
        # 8. Tikriname modelį VALIDAVIMO duomenyse
        # ========================================================

        p_val = grandine.predict_proba(X_val)[:, 1]

        # ========================================================
        # 9. Apskaičiuojame PR-AUC
        # ========================================================

        pr_auc = _pr_auc(y_val, p_val)

        # ========================================================
        # 10. Jeigu šis modelis geresnis – išsaugome
        # ========================================================

        if pr_auc > geriausias_pr_auc:
            geriausias_pr_auc = pr_auc
            geriausia_grandine = grandine
            geriausi_parametrai = params

    # ============================================================
    # 11. Grąžiname geriausią boostingo variantą
    # ============================================================

    return TrainedModel(
        name="XGBoost",
        kind="boosting",
        pipeline=geriausia_grandine,
        best_params=geriausi_parametrai,
        val_pr_auc=geriausias_pr_auc,
        notes=(
            "Pagrindinis intelektualus metodas: XGBoost tipo gradientinis boostingas, "
            "parašytas rankiniu būdu (src/rankiniai_modeliai.py, klasė "
            "GradientinisBoostingas) – logistinės nuostolių funkcijos gradientas ir "
            "Hessian, medžiai iš histogramų, lapo reikšmė w = -ΣG/(ΣH+λ). Galutinė "
            "tikimybė p̂ = σ(F(x)), F(x) = F₀ + η·Σ_k medis_k(x)."
        ),
        extra={"scale_pos_weight": scale_pos_weight},
    )


# ============================================================
# 5. Formulės patikra
# ============================================================


def logistic_formula_check(model: TrainedModel, X: pd.DataFrame, n: int = 5) -> dict:
    """Tikrina, kad modelio tikimybė tikrai yra p̂ = σ(β₀ + Xβ).

    `linear_score` ir `logistic_probability` (src/formulas.py) skaičiuojamos
    nepriklausomai nuo mokymo kodo, todėl sutapimas rodo, kad formulė
    ataskaitoje atitinka tai, kas realiai vykdoma.
    """
    if model.kind != "logistic":
        raise ValueError("Formulės patikra skirta logistinei regresijai.")
    grandine = model.pipeline
    clf = grandine.named_steps["clf"]
    X_trans = grandine.transform(X.iloc[:n])
    z = linear_score(X_trans, float(clf.intercept_), clf.coef_)
    p_formula = logistic_probability(z)
    p_modelio = model.predict_proba(X.iloc[:n])
    return {
        "z": z.tolist(),
        "p_formula": p_formula.tolist(),
        "p_sklearn": p_modelio.tolist(),
        "max_abs_diff": float(np.max(np.abs(p_formula - p_modelio))),
    }
