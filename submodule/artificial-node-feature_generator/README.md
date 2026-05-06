# artificial-node-feature_generator

Standalone utilities for rewriting node features after a graph dataset has already
been loaded into a PyG-style `data` object.

## Install

```bash
git clone --recurse-submodules https://github.com/Hzw-ZuttoZutto/artificial-node-feature_generator.git
cd artificial-node-feature_generator
pip install -e .
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

## Use As A Source Checkout

This repository can also be used directly as a Git submodule without installing
it as a Python package.

```python
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
submodule_root = repo_root / "submodule" / "artificial-node-feature_generator"
sys.path.insert(0, str(submodule_root))

from feature_rewrite import load_repo_local_feature_rewrite_api

rewrite_features = load_repo_local_feature_rewrite_api().rewrite_features
```

## Usage

```python
from artificial_node_feature_generator import rewrite_features

data = load_dataset(...)
rewritten = rewrite_features(
    data,
    feature="node_degree",
    params={"feature_dim": 128},
    seed=7,
)
```

`rewrite_features(...)` always returns a rewritten copy of `data` with `data.x`
replaced. Internal cache files are stored under the repository-local `.cache/`
directory and are reused automatically for the same graph, feature mode,
parameters, and seed.

## Supported Features

- `raw`
- `random_normal`
- `shared`
- `node_degree`
- `degree_bucket_range`
- `degree_bucket_distribution`
- `pagerank`
- `eigen`
- `eigen_norm`
- `deepwalk`

## Notes

- `node_degree` first computes degree categories and then applies a random,
  frozen embedding lookup to map them into `feature_dim`.
- `degree_bucket_range` and `degree_bucket_distribution` first bucketize node
  degrees and then apply a random, frozen embedding lookup to map them into
  `feature_dim`.
- `pagerank` computes one scalar score per node and repeats it across all
  `feature_dim` columns.
- `eigen` and `eigen_norm` compute fixed spectral features directly.
- `deepwalk` is generated inside this repository through the bundled `deepwalk`
  git submodule and then cached locally.
- Exposed `deepwalk` parameters are:
  `feature_dim`, `walk_length`, `number_walks`, `window_size`, `workers`,
  `undirected`.
- Default `deepwalk` parameters match the legacy CLI defaults:
  `number_walks=10`, `walk_length=40`, `window_size=5`, `workers=1`,
  `undirected=True`.
- The package works best with `torch_geometric.data.Data`, but any object with
  `x`, `num_nodes`, and either `adj_t` or `edge_index` is supported.
