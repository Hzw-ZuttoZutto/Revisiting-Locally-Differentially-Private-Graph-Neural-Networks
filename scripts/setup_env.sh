#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# torch-sparse 0.6.18 is supported on Python 3.10/3.11, not Python 3.13.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  SELECTED_PYTHON="$PYTHON_BIN"
else
  SELECTED_PYTHON=""
  for candidate in python3.10 python3.11 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    version="$($candidate -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$version" == "3.10" || "$version" == "3.11" ]]; then
      SELECTED_PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "$SELECTED_PYTHON" ]]; then
  echo "ERROR: Python 3.10 or 3.11 is required; Python 3.13 is unsupported by torch-sparse 0.6.18." >&2
  echo "Set PYTHON_BIN to a Python 3.10/3.11 interpreter or load a matching module/conda environment." >&2
  exit 2
fi
SELECTED_VERSION="$($SELECTED_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$SELECTED_VERSION" != "3.10" && "$SELECTED_VERSION" != "3.11" ]]; then
  echo "ERROR: PYTHON_BIN=$SELECTED_PYTHON resolves to Python $SELECTED_VERSION; use Python 3.10 or 3.11." >&2
  exit 2
fi

VENV_ROOT="${AEC_VENV:-$REPO_ROOT/.venv}"
"$SELECTED_PYTHON" -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip

if python - <<'PY' >/dev/null 2>&1
import torch, torch_geometric, torch_sparse
PY
then
  echo "Using the existing CUDA-matched torch/PyG stack."
else
  echo "Installing the core torch/PyG stack."
  python -m pip install "torch>=2.8,<2.9"
  python -m pip install --no-deps "torch-geometric==2.8.0"
  TORCH_TAG="$(python -c 'import torch; print(torch.__version__)')"
  if [[ "$TORCH_TAG" != *+* ]]; then
    TORCH_TAG="${TORCH_TAG}+cpu"
  fi
  PYG_WHL_URL="${PYG_WHL_URL:-https://data.pyg.org/whl/torch-${TORCH_TAG}.html}"
  echo "Installing torch-sparse from: $PYG_WHL_URL"
  python -m pip install --only-binary=:all: --find-links "$PYG_WHL_URL" "torch-sparse>=0.6.18,<0.6.19"
  TEMP_REQUIREMENTS="$(mktemp)"
  trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT
  grep -Ev '^(torch|torch-geometric|torch-sparse)([<>=]|$)' "$REPO_ROOT/requirements.txt" > "$TEMP_REQUIREMENTS"
  python -m pip install -r "$TEMP_REQUIREMENTS"
fi

python -m pip install 'nbformat>=5.10,<6' 'nbclient>=0.10,<1' 'jupyter>=1.0,<2' 'ipykernel>=6.29,<7'
python -m ipykernel install --user --name rethinking-dp-gnn-aec --display-name "Rethinking DP-GNN AEC"
echo "Environment ready. Activate with: source $VENV_ROOT/bin/activate"
