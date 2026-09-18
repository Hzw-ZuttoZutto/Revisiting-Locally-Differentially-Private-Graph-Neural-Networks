#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv "${AEC_VENV:-$REPO_ROOT/.venv}"
source "${AEC_VENV:-$REPO_ROOT/.venv}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements-aec.txt"
python -m ipykernel install --user --name rethinking-dp-gnn-aec --display-name "Rethinking DP-GNN AEC"
echo "Environment ready. Activate with: source ${AEC_VENV:-$REPO_ROOT/.venv}/bin/activate"
