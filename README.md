## Quick start

Create the isolated Conda environment and install all dependencies:

```bash
bash scripts/setup_env.sh
conda activate rethinking-dp-gnn-aec
```

The setup script creates the environment with Python 3.10 and installs the PyTorch, PyG, notebook, and experiment dependencies. Then open the notebooks:

```bash
python -m jupyter notebook notebooks/
```

## Two ways to run the notebooks

### Run manually in Jupyter

Open the notebook in Jupyter and run its configuration cell first, followed by the setup cell and the figure/table cells one by one. This is useful when you want to inspect each generated figure or change the GPU settings between runs.

```bash
python -m jupyter notebook notebooks/
```

### Run automatically from the command line

Use `nbconvert` to execute a notebook from top to bottom without opening the notebook UI. The executed notebook is written back to the same path.

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_1_direct.ipynb
```

Before running Notebook 2, set `AEC_EXECUTE = True` in its configuration cell if you want to start the fixed-hyperparameter reruns. Then execute the notebook:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_2_fixed_hparams.ipynb
```

For Notebook 3, set `AEC_EXECUTE = True` and choose `AEC_MODE = "scaled"` or `AEC_MODE = "full"` in its configuration cell, then execute:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_3_full_search.ipynb
```

## Which notebook to run

### Notebook 1: Direct results

Notebook 1 is the fastest entry point. It uses the experiment result data that we have already produced and directly generates the same figures and the corresponding tables shown in the paper. It does not run training or hyperparameter search.

Open `notebooks/notebook_1_direct.ipynb` from the Jupyter interface.

### Notebook 2: Fixed hyperparameters

Notebook 2 uses the hyperparameters that we have already searched. It reruns the data points needed by the paper and then generates the same figures and corresponding tables.

Edit the configuration cell to set `AEC_EXECUTE = True` when running the experiments. With `False`, the notebook only prints the execution plan. GPU selection and per-GPU concurrency are configured in the same cell.

Open `notebooks/notebook_2_fixed_hparams.ipynb` from the Jupyter interface and run the configuration cell first.

### Notebook 3: Hyperparameter search

Notebook 3 runs our hyperparameter search scripts and has two modes:

- **claim-coverage scaled mode** includes the core results needed to support the experimental conclusions.
- **full mode** runs the complete experiment configuration and is the complete reproduction path.

Both modes run the same YAML-driven hyperparameter search pipeline used by our experiments. They produce the hyperparameters needed for the data points rerun by Notebook 2, and then generate the corresponding figures and tables.

The default mode is `claim-coverage scaled`. Set `AEC_MODE = "full"` in the configuration cell when manually running the complete search.

Each notebook displays the generated PNG figures and Table 4/Table 6 directly in the notebook.
