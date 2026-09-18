#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_ROOT="${AEC_VENV:-$REPO_ROOT/.venv}"
"$PYTHON_BIN" -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip
if python - <<'PY' >/dev/null 2>&1
import torch, torch_geometric, torch_sparse
PY
then
  echo "Using the existing CUDA-matched torch/PyG stack."
  python -m pip install 'nbformat>=5.10,<6' 'nbclient>=0.10,<1' 'jupyter>=1.0,<2' 'ipykernel>=6.29,<7'
else
  echo "Installing the core and notebook dependencies declared in requirements.txt."
  python -m pip install -r "$REPO_ROOT/requirements.txt"
fi
python -m ipykernel install --user --name rethinking-dp-gnn-aec --display-name "Rethinking DP-GNN AEC"
echo "Environment ready. Activate with: source $VENV_ROOT/bin/activate"
