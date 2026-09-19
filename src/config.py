"""Fiksuoti eksperimento parametrai.

Visi modeliai naudoja tą pačią atsitiktinę sėklą ir tą patį 70/15/15 skaidymą.
Hiperparametrai ir slenkstis parenkami tik validavimo imtyje.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
TABLES_DIR = ARTIFACTS_DIR / "tables"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Klaidų kainos: FN (praleistas default) brangesnis už FP (nepagrįstas ribojimas).
C_FN = 5.0
C_FP = 1.0

TARGET_COL = "default"
ID_COL = "ID"

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
DEMOGRAPHIC_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAY_STATUS_FEATURES = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERIC_FEATURES = (
    ["LIMIT_BAL", "AGE"] + PAY_STATUS_FEATURES + BILL_FEATURES + PAY_AMT_FEATURES
)

UCI_DATASET_ID = 350
UCI_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
UCI_DOI = "10.24432/C55S3H"

SEX_MAP = {1: "vyras", 2: "moteris"}
EDUCATION_MAP = {
    1: "magistrantūra / aukštesnysis",
    2: "universitetas",
    3: "vidurinis",
    4: "kita / nežinoma",
}
MARRIAGE_MAP = {1: "vedęs / ištekėjusi", 2: "nevedęs", 3: "kita / nežinoma"}
