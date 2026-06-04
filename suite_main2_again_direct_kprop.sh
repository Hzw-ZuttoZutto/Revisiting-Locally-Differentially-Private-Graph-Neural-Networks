#!/usr/bin/env bash
set -euo pipefail

export ARTIFICIAL_NODE_FEATURE_CACHE_ROOT=/data/hzw/Rethinking_DP_GNN_runtime/cache/artificial-node-feature-generator-cache

python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/main2_again/sage/direct_kprop.yaml --output_root_dir paper_experiments/main2_again/sage/direct_kprop.yaml
python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/main2_again/gcn/direct_kprop.yaml --output_root_dir paper_experiments/main2_again/gcn/direct_kprop.yaml
python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/main2_again/gat/direct_kprop.yaml --output_root_dir paper_experiments/main2_again/gat/direct_kprop.yaml
