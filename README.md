# Rethinking DP-GNN Artifact Evaluation

This artifact provides three reproducibility entry points for the paper figures. The notebooks are intentionally separated by the amount of computation they perform.

## Quick start

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m jupyter notebook notebooks/notebook_1_direct.ipynb
```

Notebook 1 is CPU-only and reads the compact reference tables under `scripts/aec/reference/`. It generates the eight paper figures without requiring the original multi-gigabyte experiment trees.

Notebook 2 reruns selected points with fixed hyperparameters. Set `AEC_EXECUTE=1` to execute training; otherwise it prints the planned jobs. After execution, the runner aggregates the produced CSVs into the figure/table input before rendering. Set `AEC_GPU_IDS` and `AEC_MAX_PARALLEL_PER_GPU` for the target machine. The default is one visible GPU and one concurrent job per GPU.

Notebook 3 keeps the full search configuration path and defaults to the claim-coverage scaled mode. Set `AEC_MODE=full` to inspect or run the full configuration expansion. When search execution is enabled, completed manifests are matched back to the compact figure tables before rendering. Full mode is intended for authors with sufficient compute; the scaled mode is the AEC evaluation path.

## Figure and claim map

The exact datasets, methods, axes, and scaled coverage are declared in `scripts/aec/claim_coverage.yaml`. Figure cells display PNGs inline; generated PDF/CSV/LaTeX files are not written by default.

The compact reference tables preserve the plotted statistics and ten-seed rows used to compute the displayed summaries. The same notebooks also generate Table 4 and Table 6. Their settings are emitted with the implementation labels `FeatFree-P`, `FeatFree-Kprop`, and `FeatFree-HOA`. The original lab result trees are not required by the notebooks.

## Portability

All artifact paths are resolved relative to the repository root. The notebooks do not depend on `/data/hzw`, `/home/hzw`, or repository symlinks. Dataset acquisition and CUDA availability are checked at runtime; no experiment is started automatically by opening Notebook 1.

## AEC expectations

The direct notebook demonstrates functionality and exact figure generation. The fixed-hyperparameter notebook and the scaled search notebook provide independent reruns for the main trends. Full search is retained as a transparent author path and is not the default evaluator path because the complete search is substantially longer than a one-day evaluation budget.
