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

Open the notebook in Jupyter and run the setup cell followed by the figure/table cells one by one. This is useful when you want to inspect each generated figure or change the GPU settings between runs.

```bash
python -m jupyter notebook notebooks/
```

For Notebook 2 and Notebook 3, set the execution variables before opening Jupyter:

```bash
# Actually run the training/search commands.
export AEC_EXECUTE=1
# Comma-separated GPU IDs available to the notebook.
export AEC_GPU_IDS=0
# Maximum number of concurrent jobs assigned to each GPU.
export AEC_MAX_PARALLEL_PER_GPU=1
# Use the claim-coverage scaled search mode for Notebook 3.
export AEC_MODE=scaled
# Open all three notebooks in Jupyter for manual execution.
python -m jupyter notebook notebooks/
```

### Run automatically from the command line

Use `nbconvert` to execute a notebook from top to bottom without opening the notebook UI. The executed notebook is written back to the same path.

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_1_direct.ipynb
```

For Notebook 2:

```bash
AEC_EXECUTE=1 AEC_GPU_IDS=0 AEC_MAX_PARALLEL_PER_GPU=1 \
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_2_fixed_hparams.ipynb
```

For Notebook 3 claim-coverage scaled mode:

```bash
AEC_EXECUTE=1 AEC_MODE=scaled AEC_GPU_IDS=0 AEC_MAX_PARALLEL_PER_GPU=1 \
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_3_full_search.ipynb
```

For the complete search, replace `AEC_MODE=scaled` with `AEC_MODE=full`.

## Which notebook to run

### Notebook 1: Direct results

Notebook 1 is the fastest entry point. It uses the experiment result data that we have already produced and directly generates the same figures and the corresponding tables shown in the paper. It does not run training or hyperparameter search.

Open `notebooks/notebook_1_direct.ipynb` from the Jupyter interface.

### Notebook 2: Fixed hyperparameters

Notebook 2 uses the hyperparameters that we have already searched. It reruns the data points needed by the paper and then generates the same figures and corresponding tables.

Set `AEC_EXECUTE=1` to run the experiments. Without this variable, the notebook only prints the execution plan. GPU selection and per-GPU concurrency are controlled with `AEC_GPU_IDS` and `AEC_MAX_PARALLEL_PER_GPU`.

Open `notebooks/notebook_2_fixed_hparams.ipynb` from the Jupyter interface after setting the execution variables above.

### Notebook 3: Hyperparameter search

Notebook 3 runs our hyperparameter search scripts and has two modes:

- **claim-coverage scaled mode** includes the core results needed to support the experimental conclusions.  
- **full mode** runs the complete experiment configuration and is the complete reproduction path.           

Both modes run the same YAML-driven hyperparameter search pipeline used by our experiments. They produce the hyperparameters needed for the data points rerun by Notebook 2, and then generate the corresponding figures and tables.

The default mode is `claim-coverage scaled`. In manual mode, leave `AEC_MODE` unset or set it to `scaled`. Set `AEC_MODE=full` when manually running the complete search.

Each notebook displays the generated PNG figures and Table 4/Table 6 directly in the notebook.
