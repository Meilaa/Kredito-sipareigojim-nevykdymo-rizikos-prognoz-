# AI naudojimo žurnalas

Galutinio egzamino reikalavimas: svarbiausios užklausos, priimti ir atmesti pasiūlymai, bent dvi aptiktos AI klaidos arba nepatikrintos prielaidos ir jų patikros būdas.

## Kam AI buvo naudojamas

Cursor agentas (Grok 4.6) naudotas kodo struktūrai, modulių išdėstymui, Streamlit gynimo svetainei ir Jupyter užrašinei parengti. Citatos, DOI, duomenų aprašymas ir skaitiniai rezultatai tikrinti atskirai: DOI atidaryti pirminiame šaltinyje, kodas paleistas lokaliai, metrikos perskaičiuotos iš `artifacts/`.

## Svarbiausios užklausos

1. Sukurti galutinio egzamino sprendimą pagal Word užduotį ir kolokviumo PDF planą.
2. Sprendimą parodyti svetainėje, kad būtų galima apsiginti darbą, įskaitant kodo dalį; galima naudoti Jupyter.
3. Implicitūs tolesni patikslinimai: atkuriamas eksperimentas, formulės–kodo ryšys, abliacija, atsparumas, AI žurnalas.

## Priimti pasiūlymai

- Modulinė Python grandinė, atitinkanti kolokviumo etapus: `data` → `preprocess` → `models` → `calibrate` → `metrics` / `evaluate` → `ablation`.
- Baseline: bendras dažnis + logistinė regresija; intelektualūs metodai: XGBoost ir Random Forest.
- Stratifikuotas 70/15/15; hiperparametrai ir slenkstis tik validavime.
- Gynimo svetainė `streamlit run svetaine.py`, o ne vien Jupyter.
- Formulės išskirtos faile `src/formulas.py` ir svetainėje rodomos šalia kodo.
- Gyvas bandymas: vienas klientas ir CSV įkėlimas nematytam testui; C_FN/C_FP slankiklis slenksčiui keisti be pakartotinio mokymo.

## Atmesti pasiūlymai

- Cursor Canvas vietoje svetainės – tai ne gynimo demonstracija naršyklėje.
- Tik Jupyter be svetainės – sunkiau gynime šokinėti tarp formulių, grafikų ir gyvo bandymo.
- MLP kaip trečias intelektualus metodas šiame paleidime – plane jis buvo alternatyva; du intelektualūs metodai užtenka, o MLP jautresnis masteliui ir ilgiau mokosi.
- Nenaudoti demografinių požymių visai – teisingumo analizei jie palikti, o poveikis tikrinamas abliacija „be demografijos“.

## Aptiktos AI klaidos / nepatikrintos prielaidos

### 1. UCI stulpelių vardai

**Prielaida.** `ucimlrepo` (id=350) grąžina stulpelius `LIMIT_BAL`, `SEX`, `PAY_0` ir t. t.  
**Kas nutiko.** Pirmas paleidimas baigėsi klaida: logistinė regresija gavo 0 požymių, nes rinkinys turėjo `X1`–`X23`.  
**Patikra.** `df.columns` ir pirmos eilutės palygintos su Yeh ir Lien (2009) aprašu; `X1=20000`, `X2=2`, `X6=2` atitinka `LIMIT_BAL`, `SEX`, `PAY_0`. Įdėtas žemėlapis `UCI_X_MAP` faile `src/data.py`.

### 2. XGBoost macOS aplinkoje

**Prielaida.** `pip install xgboost` pakanka.  
**Kas nutiko.** `libxgboost.dylib` nerado `libomp.dylib`.  
**Patikra.** Pakartotas `from xgboost import XGBClassifier` po `libomp` įrašymo į `xgboost/lib`. Skriptas: `scripts/ensure_libomp.sh`.

### 3. Literatūros metrikų perkėlimas

**Prielaida.** Straipsnyje [2] minimos XGBoost AUC ≈ 0,95 ir PR-AUC ≈ 0,89 gali būti laukiamos ir šiame UCI rinkinyje.  
**Patikra.** DOI 10.54254/2754-1169/2025.BL29296 atidarytas; tai kitas viešas kredito rinkinys, ne būtinai UCI 30 000 klientų. Yeh ir Lien (2009) bei pakartotiniai UCI eksperimentai paprastai duoda ROC-AUC apie 0,75–0,80 ir gerokai žemesnį PR-AUC. Šio darbo skaičiai imami tik iš `python run_experiment.py`, ne iš straipsnio lentelės.

## DOI patikra

| Nr. | DOI | Statusas |
|---|---|---|
| [1] | 10.1016/j.eswa.2007.12.020 | patvirtinta; UCI DOI 10.24432/C55S3H |
| [2] | 10.54254/2754-1169/2025.BL29296 | patvirtinta (Wu, 2025, AEMPS 240, 70–79) |
| [3] | 10.1007/s42521-026-00202-6 | patvirtinta (Digital Finance 8, Article 35, 2026) |
| [4] | 10.4018/JOEUC.402038 | patvirtinta (JOEUC 38(1), 2026) |

Kiekviena citata tikrinta pagal DOI nuorodą, ne pagal AI perpasakojimą.
