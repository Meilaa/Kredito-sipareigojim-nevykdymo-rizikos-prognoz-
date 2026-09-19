# Kredito įsipareigojimų nevykdymo rizikos prognozė

Saugykla: [https://github.com/Meilaa/Kredito-sipareigojim-nevykdymo-rizikos-prognoz-](https://github.com/Meilaa/Kredito-sipareigojim-nevykdymo-rizikos-prognoz-)

Modelis pagal UCI *Default of Credit Card Clients* rinkinį įvertina kalibruotą tikimybę, kad klientas kitą mėnesį nevykdys finansinių įsipareigojimų. Tai sprendimų palaikymo priemonė, ne automatinis kredito verdiktas.

## Viena komanda eksperimentui

```bash
python run_experiment.py
```

Sėkla: `42`. Skaidymas: stratifikuotas **70 / 15 / 15**. Preprocessing parametrai skaičiuojami tik mokymo imtyje. Hiperparametrai, kalibratorius ir slenkstis parenkami tik validavimo imtyje.

## Svetainė gynimui

```bash
streamlit run svetaine.py
```

Naršyklėje atsidarys svetainė su problema, formulėmis, kodu, rezultatais, kalibracija, grupių analize, abliacija, klaidų pavyzdžiais ir gyvu bandymu (vienas klientas arba dėstytojo CSV).

## PDF ataskaita

`reports/Egzamino_ataskaita.pdf` – užduotis, sprendimo dokumentacija, palyginimo lentelės ir grafikai.

```bash
python scripts/make_report.py
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

macOS: XGBoost reikalauja OpenMP (`libomp`). Jei `import xgboost` nustoja veikti, paleiskite `bash scripts/ensure_libomp.sh`.

## Duomenys

Rinkinys atsisiunčiamas automatiniu būdu iš UCI (`id=350`):  
[https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
DOI: 10.24432/C55S3H  

Failo kopijos teikti nereikia.

## Modeliai


| Metodas                | Vaidmuo                           |
| ---------------------- | --------------------------------- |
| Bendras default dažnis | naivus baseline                   |
| Logistinė regresija    | interpretuojamas baseline         |
| XGBoost                | pagrindinis intelektualus metodas |
| Random Forest          | antras intelektualus metodas      |


Vertinimas: **PR-AUC**, **Brier score**, kalibracijos kreivė, klaidų kainos kreivė `Cost(t) = C_FN·FN + C_FP·FP` (numatytai `C_FN=5`, `C_FP=1`).

## Formulės kode

Žr. `src/formulas.py` ir svetainės skiltį **Formulės ir kodas**. Logistinės regresijos `p̂ = 1/(1+e^{-z})` sutapatinama su `sklearn` tikimybėmis.

## Struktūra

```
src/           duomenys, preprocessing, modeliai, formulės, metrikos
run_experiment.py
svetaine.py    gynimo svetainė
notebooks/     Jupyter užrašinė
artifacts/     lentelės, grafikai, modeliai, results.json
reports/       AI žurnalas ir PDF ataskaita
```

