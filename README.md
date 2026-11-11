# SLHMM

Core source code for SLHMM, a trajectory map-matching pipeline on OpenStreetMap
road networks. The package includes road representation learning, emission and
transition models, routing, Viterbi decoding, and evaluation.

## Setup

Use a Linux Python environment with compatible PyTorch and DGL installations.
The other dependencies are NumPy, SciPy, NetworkX, OSMnx, PyProj, Shapely, H3,
Matplotlib, and TensorBoard.

Run the commands below from the repository root. Set the project root explicitly
so that data and generated artifacts are resolved relative to this directory:

```bash
export SLHMM_PROJECT_ROOT="$PWD"
python -m slhmm.training.cli --help
```

## Data preparation

Provide a road graph and preprocessed training data under the appropriate dataset
directory:

| Dataset option | Data directory |
| --- | --- |
| `--dataset hangzhou` | `data/osm_filtered_10098/` |
| `--dataset xian` | `data/xian_val200/` |

Each directory must contain `graph/graph.graphml` and a `training/` directory with
`mappings.obj`, `alignment_train.obj`, `alignment_val.obj`,
`cell_candidates_k200.obj` (Hangzhou) or `cell_candidates_k1200.obj` (Xi'an),
`manifest.json`, and the data shards referenced by the manifest. See
[`slhmm/training/data.py`](slhmm/training/data.py) for the expected data structures.

The `prepare` command builds model artifacts from these preprocessed inputs;
conversion from raw trajectories is not included in this repository.

## Training and evaluation

For the full model on the Hangzhou dataset:

```bash
python -m slhmm.training.cli --dataset hangzhou --variant full --experiment-id hangzhou_full prepare
python -m slhmm.training.cli --dataset hangzhou --variant full --experiment-id hangzhou_full run-all
```

After preparation, `run-all` performs road pretraining, emission training,
candidate refresh, path preparation, transition training, joint fine-tuning,
and evaluation. To evaluate the trained model again:

```bash
python -m slhmm.training.cli --dataset hangzhou --variant full --experiment-id hangzhou_full evaluate
```

Use the same dataset, variant, and experiment ID across commands. Global options
must appear before the subcommand. Use `--dataset xian` with a separate experiment
ID for Xi'an, or select an ablation using `--variant` (see `--help`). Defaults and
dataset profiles are defined in
[`slhmm/configs/training.py`](slhmm/configs/training.py).

Generated features are stored in `intermediate/features/`, checkpoints in
`intermediate/checkpoints/<experiment-id>/`, and evaluation results in
`intermediate/runs/<experiment-id>/viterbi/evaluation.json`.

## Data availability

Due to privacy policies, the datasets used in this project cannot be publicly
released and are not included in this repository. This repository provides the
core source code only. Running training or evaluation requires access to data
in the expected format.
