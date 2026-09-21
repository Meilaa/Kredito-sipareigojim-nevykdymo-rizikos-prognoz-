#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Kuriama virtuali aplinka..."
  python3.11 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  bash scripts/ensure_libomp.sh || true
fi
if [[ ! -f artifacts/results.json ]]; then
  .venv/bin/python run_experiment.py
fi
exec .venv/bin/streamlit run svetaine.py
