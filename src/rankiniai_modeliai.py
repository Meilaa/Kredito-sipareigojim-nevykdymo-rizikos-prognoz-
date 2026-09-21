"""Rankiniu būdu (tik su numpy) parašyti mokymo algoritmai.

Šiame faile NĖRA jokios mokymo bibliotekos. Visi žingsniai – tikimybė,
gradientas, Hessian, medžio padalijimai, lapų reikšmės, bootstrap – parašyti
čia pat ir atitinka formules, rodomas ataskaitoje.

Turinys:
    1. sigmoide                     – p = 1 / (1 + e^{-z})
    2. LogistineRegresija           – Newton–Raphson (IRLS) optimizavimas
    3. kvantiliu_briaunos / i_binus – požymių diskretizavimas medžiams
    4. MedzioStruktura              – bendra medžio saugojimo ir prognozės dalis
    5. RegresinisMedis              – medis gradientiniam boostingui (g, h)
    6. KlasifikacinisMedis          – medis su Gini kriterijumi (miškui)
    7. AtsitiktinisMiskas           – bootstrap + požymių atranka
    8. GradientinisBoostingas       – XGBoost tipo algoritmas su Hessian'u
    9. Grandine                     – preprocessing + modelis vienoje vietoje
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. Sigmoidinė funkcija
# ============================================================


def sigmoide(z: np.ndarray) -> np.ndarray:
    """p = 1 / (1 + e^{-z}).

    np.clip apsaugo nuo skaitinio perpildymo, kai |z| labai didelis:
    e^{1000} kompiuteryje yra begalybė, o e^{500} dar skaičiuojamas.
    """
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _klasiu_svoriai(y: np.ndarray, class_weight) -> np.ndarray:
    """Kiekvienam klientui priskiria svorį s_i.

    class_weight=None      -> visi svoriai 1
    class_weight="balanced"-> s_i = N / (2 * N_{y_i}); retesnė klasė svaresnė
    """
    y = np.asarray(y).astype(int)
    svoriai = np.ones(len(y), dtype=float)
    if class_weight == "balanced":
        n = len(y)
        n_pos = max(int(np.sum(y == 1)), 1)
        n_neg = max(int(np.sum(y == 0)), 1)
        svoriai[y == 1] = n / (2.0 * n_pos)
        svoriai[y == 0] = n / (2.0 * n_neg)
    return svoriai


# ============================================================
# 2. Logistinė regresija (rankinis Newton–Raphson)
# ============================================================


class LogistineRegresija:
    """Logistinė regresija, optimizuojama Newton–Raphson metodu.

    Modelis:
        z_i = β₀ + β₁x_i1 + ... + β_p x_ip
        p_i = 1 / (1 + e^{-z_i})

    Minimizuojama funkcija (su L2 bauda, laisvasis narys nebaudžiamas):

        L(β) = Σ_i s_i · [ -y_i·log(p_i) - (1-y_i)·log(1-p_i) ]
               + (1 / (2C)) · Σ_{j≥1} β_j²

    Newton–Raphson žingsnis:

        gradientas:  g = Xᵀ (s ⊙ (p - y)) + λ·β
        Hessian:     H = Xᵀ · diag(s ⊙ p ⊙ (1-p)) · X + λ·I
        atnaujinimas: β ← β - H⁻¹ g

    Skirtingai nuo paprasto gradientinio nusileidimo, Newton metodas naudoja
    antrąsias išvestines, todėl konverguoja per ~8 iteracijas ir nereikia
    rinkti mokymo žingsnio (learning rate).
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight=None,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        self.C = float(C)
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogistineRegresija":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float)

        # --------------------------------------------------------
        # Prie požymių pridedamas vienetų stulpelis, kad β₀ (laisvasis
        # narys) būtų skaičiuojamas kartu su kitais koeficientais.
        # --------------------------------------------------------
        n, p = X.shape
        X_su_vienetais = np.hstack([np.ones((n, 1)), X])

        # Svoriai klasėms (nesubalansuotiems duomenims)
        s = _klasiu_svoriai(y, self.class_weight)

        # L2 baudos koeficientas λ = 1/C. Laisvajam nariui bauda = 0.
        lambda_ = 1.0 / self.C
        baudos_vektorius = np.full(p + 1, lambda_)
        baudos_vektorius[0] = 0.0

        beta = np.zeros(p + 1, dtype=float)

        for iteracija in range(self.max_iter):
            # 1. Dabartinė tikimybė kiekvienam klientui
            z = X_su_vienetais @ beta
            prob = sigmoide(z)

            # 2. Gradientas: g = Xᵀ(s·(p - y)) + λβ
            gradientas = X_su_vienetais.T @ (s * (prob - y)) + baudos_vektorius * beta

            # 3. Hessian: H = Xᵀ·diag(s·p·(1-p))·X + λI
            #    W yra įstrižainės matrica, todėl užtenka padauginti eilutes.
            W = s * prob * (1.0 - prob)
            H = X_su_vienetais.T @ (X_su_vienetais * W[:, None])
            H[np.diag_indices_from(H)] += baudos_vektorius

            # 4. Newton žingsnis: sprendžiame H·delta = g
            try:
                delta = np.linalg.solve(H, gradientas)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, gradientas, rcond=None)[0]

            beta = beta - delta
            self.n_iter_ = iteracija + 1

            # 5. Sustojimo kriterijus: žingsnis jau nebejuda
            if np.max(np.abs(delta)) < self.tol:
                break

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].copy()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """z = β₀ + Xβ."""
        if self.coef_ is None:
            raise RuntimeError("Modelis neapmokytas.")
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Grąžina abiejų klasių tikimybes, kaip įprasta klasifikatoriams."""
        p1 = sigmoide(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])


# ============================================================
# 3. Požymių diskretizavimas (binai) medžiams
# ============================================================


def kvantiliu_briaunos(X: np.ndarray, n_binu: int = 64) -> list[np.ndarray]:
    """Kiekvienam požymiui suranda ribas, kurios dalija jį į ~vienodas dalis.

    Vietoje visų galimų padalijimų (jų būtų iki n-1) tikriname tik iki
    n_binu-1 kandidatų. Tą patį triuką naudoja XGBoost `tree_method="hist"`.
    """
    X = np.asarray(X, dtype=float)
    briaunos = []
    kvantiliai = np.linspace(0.0, 1.0, n_binu + 1)[1:-1]
    for j in range(X.shape[1]):
        stulpelis = X[:, j]
        ribos = np.unique(np.quantile(stulpelis, kvantiliai))
        briaunos.append(ribos.astype(float))
    return briaunos


def i_binus(X: np.ndarray, briaunos: list[np.ndarray]) -> np.ndarray:
    """Pakeičia tikrą požymio reikšmę bino numeriu.

    bin(x) = kiek ribų yra <= x. Tada „bin <= b" yra tas pats, kas
    „x < briaunos[b]", todėl mokymas ir prognozė sutampa tiksliai.
    """
    X = np.asarray(X, dtype=float)
    binai = np.empty(X.shape, dtype=np.int16)
    for j, ribos in enumerate(briaunos):
        binai[:, j] = np.searchsorted(ribos, X[:, j], side="right")
    return binai


# ============================================================
# 4. Medžio struktūra ir prognozė
# ============================================================


class MedzioStruktura:
    """Medis saugomas lygiaverčiuose masyvuose, ne objektuose.

    Kiekvienam mazgui i:
        pozymis[i]   – pagal kurį požymį dalijama (-1, jei lapas)
        riba[i]      – tikra požymio reikšmė: kairė, kai x < riba
        kaire[i]     – kairio vaiko indeksas
        desine[i]    – dešinio vaiko indeksas
        reiksme[i]   – lapo reikšmė (boostinge – w, miške – tikimybė)
    """

    def __init__(self) -> None:
        self.pozymis: list[int] = []
        self.riba: list[float] = []
        self.kaire: list[int] = []
        self.desine: list[int] = []
        self.reiksme: list[float] = []

    def naujas_mazgas(self) -> int:
        self.pozymis.append(-1)
        self.riba.append(0.0)
        self.kaire.append(-1)
        self.desine.append(-1)
        self.reiksme.append(0.0)
        return len(self.pozymis) - 1

    def uzbaigti(self) -> None:
        """Sąrašus paverčia numpy masyvais, kad prognozė būtų greita."""
        self.pozymis = np.asarray(self.pozymis, dtype=np.int32)
        self.riba = np.asarray(self.riba, dtype=float)
        self.kaire = np.asarray(self.kaire, dtype=np.int32)
        self.desine = np.asarray(self.desine, dtype=np.int32)
        self.reiksme = np.asarray(self.reiksme, dtype=float)

    def prognozuoti(self, X: np.ndarray) -> np.ndarray:
        """Visus klientus vienu metu „nuleidžia" medžiu iki lapų.

        Ciklas kartojasi tik tiek kartų, kokio gylio yra medis,
        o ne tiek kartų, kiek yra klientų.
        """
        X = np.asarray(X, dtype=float)
        mazgas = np.zeros(len(X), dtype=np.int32)
        while True:
            yra_lapas = self.pozymis[mazgas] == -1
            if np.all(yra_lapas):
                break
            eiles = np.flatnonzero(~yra_lapas)
            dabartinis = mazgas[eiles]
            f = self.pozymis[dabartinis]
            t = self.riba[dabartinis]
            i_kaire = X[eiles, f] < t
            mazgas[eiles] = np.where(i_kaire, self.kaire[dabartinis], self.desine[dabartinis])
        return self.reiksme[mazgas]


# ============================================================
# 5. Regresinis medis gradientiniam boostingui
# ============================================================


class RegresinisMedis:
    """Medis, kuris mokomas ne iš klasių, o iš gradientų g ir Hessian'ų h.

    Lapo reikšmė (XGBoost formulė):

                  Σ g_i
        w = - ---------------
               Σ h_i + lambda

    Padalijimo nauda:

        gain = ½ · [ G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ) ] - gamma
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.medis = MedzioStruktura()

    def _lapo_reiksme(self, G: float, H: float) -> float:
        return -G / (H + self.reg_lambda)

    def _balas(self, G: float, H: float) -> float:
        """G²/(H+λ) – kiek „geras" yra mazgas."""
        return (G * G) / (H + self.reg_lambda)

    def fit(
        self,
        binai: np.ndarray,
        briaunos: list[np.ndarray],
        g: np.ndarray,
        h: np.ndarray,
        eiles: np.ndarray,
        pozymiai: np.ndarray,
        n_binu: int,
    ) -> "RegresinisMedis":
        self._briaunos = briaunos
        self._n_binu = n_binu
        saknis = self.medis.naujas_mazgas()
        self._auginti(saknis, binai, g, h, eiles, pozymiai, gylis=0)
        self.medis.uzbaigti()
        return self

    def _auginti(
        self,
        mazgas: int,
        binai: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        eiles: np.ndarray,
        pozymiai: np.ndarray,
        gylis: int,
    ) -> None:
        G = float(np.sum(g[eiles]))
        H = float(np.sum(h[eiles]))

        # Iš anksto užrašome mazgą kaip lapą; jei rasime padalijimą, perrašysime.
        self.medis.reiksme[mazgas] = self._lapo_reiksme(G, H)

        if gylis >= self.max_depth or len(eiles) < 2:
            return

        tevo_balas = self._balas(G, H)
        geriausia_nauda = 0.0
        geriausias_pozymis = -1
        geriausias_binas = -1

        for f in pozymiai:
            stulpelis = binai[eiles, f]

            # Histogramos: kiek gradiento ir Hessian'o susideda kiekviename bine
            G_hist = np.bincount(stulpelis, weights=g[eiles], minlength=self._n_binu)
            H_hist = np.bincount(stulpelis, weights=h[eiles], minlength=self._n_binu)

            # Sumos „iki bino b" – tai kairė padalijimo pusė
            G_kaire = np.cumsum(G_hist)[:-1]
            H_kaire = np.cumsum(H_hist)[:-1]
            G_desine = G - G_kaire
            H_desine = H - H_kaire

            # Per mažų vaikų nepriimame (min_child_weight = Hessian'ų suma)
            galima = (H_kaire >= self.min_child_weight) & (H_desine >= self.min_child_weight)
            if not np.any(galima):
                continue

            nauda = 0.5 * (
                (G_kaire * G_kaire) / (H_kaire + self.reg_lambda)
                + (G_desine * G_desine) / (H_desine + self.reg_lambda)
                - tevo_balas
            ) - self.gamma
            nauda = np.where(galima, nauda, -np.inf)

            b = int(np.argmax(nauda))
            if nauda[b] > geriausia_nauda:
                geriausia_nauda = float(nauda[b])
                geriausias_pozymis = int(f)
                geriausias_binas = b

        if geriausias_pozymis < 0:
            return

        ribos = self._briaunos[geriausias_pozymis]
        if geriausias_binas >= len(ribos):
            return
        riba = float(ribos[geriausias_binas])

        i_kaire = binai[eiles, geriausias_pozymis] <= geriausias_binas
        eiles_kaire = eiles[i_kaire]
        eiles_desine = eiles[~i_kaire]
        if len(eiles_kaire) == 0 or len(eiles_desine) == 0:
            return

        kairys = self.medis.naujas_mazgas()
        desinys = self.medis.naujas_mazgas()
        self.medis.pozymis[mazgas] = geriausias_pozymis
        self.medis.riba[mazgas] = riba
        self.medis.kaire[mazgas] = kairys
        self.medis.desine[mazgas] = desinys

        self._auginti(kairys, binai, g, h, eiles_kaire, pozymiai, gylis + 1)
        self._auginti(desinys, binai, g, h, eiles_desine, pozymiai, gylis + 1)

    def prognozuoti(self, X: np.ndarray) -> np.ndarray:
        return self.medis.prognozuoti(X)


# ============================================================
# 6. Klasifikacinis medis su Gini kriterijumi
# ============================================================


class KlasifikacinisMedis:
    """Medis, kuris dalija duomenis taip, kad klasės kuo labiau atsiskirtų.

    Gini nešvarumas mazge:

        Gini = 1 - p₁² - p₀²,   p₁ = svorių dalis, kur y = 1

    Padalijimo kokybė – svertinis vaikų Gini vidurkis; renkame mažiausią.
    """

    def __init__(
        self,
        max_depth: int | None = 8,
        min_samples_leaf: int = 5,
        max_features: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.medis = MedzioStruktura()

    def fit(
        self,
        binai: np.ndarray,
        briaunos: list[np.ndarray],
        y: np.ndarray,
        svoriai: np.ndarray,
        eiles: np.ndarray,
        n_binu: int,
    ) -> "KlasifikacinisMedis":
        self._briaunos = briaunos
        self._n_binu = n_binu
        self._n_pozymiu = binai.shape[1]
        saknis = self.medis.naujas_mazgas()
        self._auginti(saknis, binai, y, svoriai, eiles, gylis=0)
        self.medis.uzbaigti()
        return self

    def _pozymiu_atranka(self) -> np.ndarray:
        """Kiekviename mazge tikrinamas tik atsitiktinis požymių poaibis.

        Tai daro medžius nepriklausomesnius vienas nuo kito – pagrindinė
        atsitiktinio miško idėja.
        """
        if self.max_features is None or self.max_features >= self._n_pozymiu:
            return np.arange(self._n_pozymiu)
        return self.rng.choice(self._n_pozymiu, size=self.max_features, replace=False)

    def _auginti(
        self,
        mazgas: int,
        binai: np.ndarray,
        y: np.ndarray,
        svoriai: np.ndarray,
        eiles: np.ndarray,
        gylis: int,
    ) -> None:
        w = svoriai[eiles]
        w_pos = float(np.sum(w * y[eiles]))
        w_visi = float(np.sum(w))
        p1 = w_pos / w_visi if w_visi > 0 else 0.0

        # Lapo prognozė – svertinė default klientų dalis
        self.medis.reiksme[mazgas] = p1

        per_gilu = self.max_depth is not None and gylis >= self.max_depth
        if per_gilu or len(eiles) < 2 * self.min_samples_leaf or p1 in (0.0, 1.0):
            return

        geriausias_gini = 1.0 - (p1 * p1 + (1.0 - p1) * (1.0 - p1))
        geriausias_pozymis = -1
        geriausias_binas = -1

        for f in self._pozymiu_atranka():
            stulpelis = binai[eiles, f]

            # Svoriai pagal klases kiekviename bine
            pos_hist = np.bincount(stulpelis, weights=w * y[eiles], minlength=self._n_binu)
            vis_hist = np.bincount(stulpelis, weights=w, minlength=self._n_binu)
            kiek_hist = np.bincount(stulpelis, minlength=self._n_binu)

            pos_kaire = np.cumsum(pos_hist)[:-1]
            vis_kaire = np.cumsum(vis_hist)[:-1]
            kiek_kaire = np.cumsum(kiek_hist)[:-1]

            pos_desine = w_pos - pos_kaire
            vis_desine = w_visi - vis_kaire
            kiek_desine = len(eiles) - kiek_kaire

            galima = (kiek_kaire >= self.min_samples_leaf) & (kiek_desine >= self.min_samples_leaf)
            galima &= (vis_kaire > 0) & (vis_desine > 0)
            if not np.any(galima):
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                p_kaire = np.where(vis_kaire > 0, pos_kaire / vis_kaire, 0.0)
                p_desine = np.where(vis_desine > 0, pos_desine / vis_desine, 0.0)
            gini_kaire = 1.0 - (p_kaire**2 + (1.0 - p_kaire) ** 2)
            gini_desine = 1.0 - (p_desine**2 + (1.0 - p_desine) ** 2)

            # Svertinis vaikų nešvarumas
            gini = (vis_kaire * gini_kaire + vis_desine * gini_desine) / w_visi
            gini = np.where(galima, gini, np.inf)

            b = int(np.argmin(gini))
            if gini[b] < geriausias_gini - 1e-12:
                geriausias_gini = float(gini[b])
                geriausias_pozymis = int(f)
                geriausias_binas = b

        if geriausias_pozymis < 0:
            return

        ribos = self._briaunos[geriausias_pozymis]
        if geriausias_binas >= len(ribos):
            return
        riba = float(ribos[geriausias_binas])

        i_kaire = binai[eiles, geriausias_pozymis] <= geriausias_binas
        eiles_kaire = eiles[i_kaire]
        eiles_desine = eiles[~i_kaire]
        if len(eiles_kaire) == 0 or len(eiles_desine) == 0:
            return

        kairys = self.medis.naujas_mazgas()
        desinys = self.medis.naujas_mazgas()
        self.medis.pozymis[mazgas] = geriausias_pozymis
        self.medis.riba[mazgas] = riba
        self.medis.kaire[mazgas] = kairys
        self.medis.desine[mazgas] = desinys

        self._auginti(kairys, binai, y, svoriai, eiles_kaire, gylis + 1)
        self._auginti(desinys, binai, y, svoriai, eiles_desine, gylis + 1)

    def prognozuoti(self, X: np.ndarray) -> np.ndarray:
        return self.medis.prognozuoti(X)


# ============================================================
# 7. Atsitiktinis miškas
# ============================================================


class AtsitiktinisMiskas:
    """Daug medžių, kiekvienas mokomas kitame duomenų pavyzdyje.

    1. Bootstrap: atsitiktinai su pakartojimu paimame N klientų.
    2. Kiekviename mazge tikriname tik dalį požymių (max_features).
    3. Galutinė tikimybė – visų medžių prognozių vidurkis:

           p(x) = (1/K) · Σ_k medis_k(x)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 8,
        min_samples_leaf: int = 5,
        max_features: str | int | None = "sqrt",
        class_weight=None,
        n_binu: int = 64,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_binu = int(n_binu)
        self.random_state = int(random_state)
        self.medziai: list[KlasifikacinisMedis] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AtsitiktinisMiskas":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        # Diskretizuojame vieną kartą – visi medžiai naudoja tas pačias ribas
        self.briaunos = kvantiliu_briaunos(X, self.n_binu)
        binai = i_binus(X, self.briaunos)

        if self.max_features == "sqrt":
            k_pozymiu = max(1, int(np.sqrt(d)))
        elif isinstance(self.max_features, int):
            k_pozymiu = min(self.max_features, d)
        else:
            k_pozymiu = d

        self.medziai = []
        for k in range(self.n_estimators):
            # 1. Bootstrap pavyzdys
            eiles = rng.integers(0, n, size=n)

            # 2. Svoriai skaičiuojami TAME pavyzdyje (balanced_subsample logika)
            svoriai_pavyzdzio = _klasiu_svoriai(y[eiles].astype(int), self.class_weight)
            svoriai = np.zeros(n, dtype=float)
            np.add.at(svoriai, eiles, svoriai_pavyzdzio)
            # Kad medis matytų pasikartojančius klientus kaip didesnį svorį,
            # dirbame su unikaliomis eilėmis ir sukauptais svoriais.
            unikalios = np.unique(eiles)

            medis = KlasifikacinisMedis(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=k_pozymiu,
                random_state=self.random_state + k,
            )
            medis.fit(binai, self.briaunos, y, svoriai, unikalios, self.n_binu)
            self.medziai.append(medis)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        suma = np.zeros(len(X), dtype=float)
        for medis in self.medziai:
            suma += medis.prognozuoti(X)
        p1 = suma / max(len(self.medziai), 1)
        return np.column_stack([1.0 - p1, p1])


# ============================================================
# 8. Gradientinis boostingas (XGBoost tipo)
# ============================================================


class GradientinisBoostingas:
    """Medžiai statomi vienas po kito, kiekvienas taiso ankstesnių klaidas.

    Modelis:
        F(x) = F₀ + η·f₁(x) + η·f₂(x) + ... + η·f_K(x)
        p(x) = 1 / (1 + e^{-F(x)})

    Kiekvienoje iteracijoje:
        1. p = sigmoide(F)
        2. gradientas   g_i = s_i · (p_i - y_i)
        3. Hessian      h_i = s_i · p_i · (1 - p_i)
        4. naujas medis mokomas iš g ir h (ne iš y!)
        5. F ← F + η · medis(x)
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_child_weight: float = 5.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        scale_pos_weight: float = 1.0,
        n_binu: int = 64,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.scale_pos_weight = float(scale_pos_weight)
        self.n_binu = int(n_binu)
        self.random_state = int(random_state)
        self.medziai: list[RegresinisMedis] = []
        self.F0: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientinisBoostingas":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        # Retesnės klasės svoris – kaip XGBoost scale_pos_weight
        s = np.where(y == 1, self.scale_pos_weight, 1.0)

        # Pradinis balas: svertinis logit'as, kad startuotume nuo bazinio dažnio
        w_pos = float(np.sum(s * y))
        w_neg = float(np.sum(s * (1.0 - y)))
        p0 = w_pos / max(w_pos + w_neg, 1e-12)
        p0 = min(max(p0, 1e-6), 1 - 1e-6)
        self.F0 = float(np.log(p0 / (1.0 - p0)))

        self.briaunos = kvantiliu_briaunos(X, self.n_binu)
        binai = i_binus(X, self.briaunos)

        F = np.full(n, self.F0, dtype=float)
        k_pozymiu = max(1, int(round(self.colsample_bytree * d)))
        n_eiliu = max(1, int(round(self.subsample * n)))

        self.medziai = []
        for k in range(self.n_estimators):
            # 1. Dabartinės tikimybės
            prob = sigmoide(F)

            # 2. Pirmoji ir antroji logistinės nuostolių funkcijos išvestinės
            g = s * (prob - y)
            h = s * prob * (1.0 - prob)

            # 3. Atsitiktinis klientų ir požymių poaibis šiam medžiui
            eiles = rng.choice(n, size=n_eiliu, replace=False) if n_eiliu < n else np.arange(n)
            pozymiai = (
                rng.choice(d, size=k_pozymiu, replace=False) if k_pozymiu < d else np.arange(d)
            )

            # 4. Medis mokomas iš gradientų, ne iš klasių
            medis = RegresinisMedis(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
            )
            medis.fit(binai, self.briaunos, g, h, eiles, pozymiai, self.n_binu)

            # 5. Naujas medis pridedamas su mokymo žingsniu η
            F += self.learning_rate * medis.prognozuoti(X)
            self.medziai.append(medis)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """F(x) = F₀ + η·Σ medis_k(x)."""
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.F0, dtype=float)
        for medis in self.medziai:
            F += self.learning_rate * medis.prognozuoti(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = sigmoide(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])


# ============================================================
# 9. Grandinė: duomenų paruošimas + rankinis modelis
# ============================================================


class Grandine:
    """Sujungia požymių kodavimą ir rankinį modelį į vieną objektą.

    Preprocessing (kategorijų kodavimas, standartizavimas) yra techninis
    žingsnis, jis mokomas TIK iš mokymo imties. Pats mokymo algoritmas –
    šio failo klasės.
    """

    def __init__(self, preprocessor, modelis) -> None:
        self.preprocessor = preprocessor
        self.modelis = modelis

    @property
    def named_steps(self) -> dict:
        return {"prep": self.preprocessor, "clf": self.modelis}

    def fit(self, X, y) -> "Grandine":
        X_paruosta = self.preprocessor.fit_transform(X)
        self.modelis.fit(np.asarray(X_paruosta, dtype=float), np.asarray(y).astype(float))
        return self

    def transform(self, X) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(X), dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        return self.modelis.predict_proba(self.transform(X))
