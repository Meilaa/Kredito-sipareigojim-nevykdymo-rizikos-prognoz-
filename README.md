# Kredito įsipareigojimų nevykdymo rizikos prognozė

Modelis pagal UCI *Default of Credit Card Clients* rinkinį įvertina kalibruotą tikimybę, kad klientas kitą mėnesį nevykdys finansinių įsipareigojimų. Tai sprendimų palaikymo priemonė, ne automatinis kredito verdiktas.

## Viena komanda eksperimentui

```bash
python run_experiment.py
```

Sėkla: `42`. Skaidymas: stratifikuotas **70 / 15 / 15**. Preprocessing parametrai skaičiuojami tik mokymo imtyje. Hiperparametrai, kalibratorius ir slenkstis parenkami tik validavimo imtyje.

## Svetainė 

```bash
streamlit run svetaine.py
```

Naršyklėje atsidarys svetainė su problema, formulėmis, kodu, rezultatais, kalibracija, grupių analize, abliacija, klaidų pavyzdžiais ir gyvu bandymu (vienas klientas arba dėstytojo CSV).

## Ataskaita (PDF ir Word)

`reports/Egzamino_ataskaita.pdf` ir `reports/Egzamino_ataskaita.docx` – užduotis, sprendimo dokumentacija, palyginimo lentelės ir grafikai.

```bash
python scripts/make_report.py
```

Tik Word:

```bash
python scripts/make_word_report.py
```

```bash
jupyter notebook notebooks/egzamino_sprendimas.ipynb
```



## Diegimas

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
streamlit run svetaine.py
```

macOS: patikrai naudojamas XGBoost reikalauja OpenMP (`libomp`). Jei `import xgboost` nustoja veikti, paleiskite `bash scripts/ensure_libomp.sh`. Pačiam eksperimentui to nereikia – rankiniams algoritmams pakanka `numpy`.

## Duomenys

Rinkinys atsisiunčiamas automatiniu būdu iš UCI (`id=350`):  
[https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
DOI: 10.24432/C55S3H  

Failo kopijos teikti nereikia.

## Modeliai


| Metodas                | Vaidmuo                           | Kaip įgyvendinta (viskas rankiniu būdu)      |
| ---------------------- | --------------------------------- | -------------------------------------------- |
| Bendras default dažnis | naivus baseline                   | mokymo imties dažnis                          |
| Logistinė regresija    | interpretuojamas baseline         | Newton–Raphson (`LogistineRegresija`)         |
| XGBoost                | pagrindinis intelektualus metodas | gradientas, Hessian, histogramų medžiai (`GradientinisBoostingas`) |
| Random Forest          | antras intelektualus metodas      | bootstrap, požymių atranka, Gini (`AtsitiktinisMiskas`) |

Lentelėse palikti įprasti metodų pavadinimai (XGBoost, Random Forest), bet **algoritmai parašyti rankiniu būdu** faile `src/rankiniai_modeliai.py` naudojant tik `numpy` – tai ne bibliotekų iškvietimas. `sklearn` naudojamas tik kategorijų kodavimui, metrikoms ir isotonic kalibravimui, `xgboost` biblioteka – tik nepriklausomai patikrai:

```bash
python scripts/patikra_rankiniai.py
```

Galutinis modelis renkamas **pagal validavimo PR-AUC**; testo imtis skaičiuojama vieną kartą kaip galutinis įvertis ir atrankoje nedalyvauja.


Vertinimas: **PR-AUC**, **Brier score**, kalibracijos kreivė, klaidų kainos kreivė `Cost(t) = C_FN·FN + C_FP·FP` (numatytai `C_FN=5`, `C_FP=1`).

## Formulės kode

Žr. `src/formulas.py`, `src/rankiniai_modeliai.py` ir svetainės skiltį **Formulės ir kodas**. Logistinės regresijos `p̂ = 1/(1+e^{-z})` sutapatinama su modelio išvestimi; boostingo lapo reikšmė `w = -ΣG/(ΣH+λ)`.

## Struktūra

```
src/           duomenys, preprocessing, rankiniai algoritmai, modelių mokymas, formulės, metrikos
run_experiment.py
svetaine.py    gynimo svetainė
notebooks/     Jupyter užrašinė
artifacts/     lentelės, grafikai, modeliai, results.json
reports/       AI žurnalas, PDF ir Word ataskaita
```

