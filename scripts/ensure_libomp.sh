#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/.venv/lib/python3.11/site-packages/xgboost/lib/libomp.dylib"
if [[ -f "$DEST" ]]; then
  echo "libomp jau yra: $DEST"
  exit 0
fi
mkdir -p "$ROOT/.deps"
curl -L -o "$ROOT/.deps/openmp.tar.gz" "https://mac.r-project.org/openmp/openmp-17.0.6-darwin20-Release.tar.gz"
tar -xzf "$ROOT/.deps/openmp.tar.gz" -C "$ROOT/.deps"
mkdir -p "$ROOT/.venv/lib/python3.11/site-packages/xgboost/lib"
cp "$ROOT/.deps/usr/local/lib/libomp.dylib" "$DEST"
echo "Įrašytas $DEST"
