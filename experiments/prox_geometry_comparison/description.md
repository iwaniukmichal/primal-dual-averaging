# Prox Geometry Comparison

Compares Euclidean and weighted Euclidean prox-functions on coordinate-imbalanced objectives. This experiment is intended to show when diagonal geometry helps or hurts dual averaging.

Outputs are written to `results/runs.csv`, `results/iterations.csv`, `results/metadata.json`, and `results/summary.csv`.

## How to run

From the repository root:

```bash
python experiments/prox_geometry_comparison/code/run.py
```

The run writes raw CSV files and creates `results/summary.csv` automatically.

Useful options:

```bash
python experiments/prox_geometry_comparison/code/run.py --dry-run
python experiments/prox_geometry_comparison/code/run.py --limit-runs 10
python experiments/prox_geometry_comparison/code/run.py --output-dir /tmp/prox_results
```

## How to change parameters

Edit `code/config.py`. The experiment matrix is built in `build_configs()`.

Important fields:

- `objective_id`: ill-conditioned and baseline nonsmooth objectives.
- `prox_fun`: `euclidean` or `weighted_euclidean`.
- `prox_weights_profile`: `uniform`, `coordinate_scale`, or `inverse_coordinate_scale`.
- `dual_averaging`: `simple` or `weighted`.
- `D`, `gamma_mult`, `restrict_to_fd`, `max_iter`, `eps`: SDA controls.

Weighted Euclidean prox requires positive coordinate weights; profiles are resolved in `experiments/_shared/objectives.py`.
