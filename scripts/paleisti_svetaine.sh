#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "Pirma sukurkite aplinką: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [[ ! -f artifacts/results.json ]]; then
  echo "Nerasti rezultatai – paleidžiamas eksperimentas."
  .venv/bin/python run_experiment.py
fi
exec .venv/bin/streamlit run svetaine.py --server.headless true
