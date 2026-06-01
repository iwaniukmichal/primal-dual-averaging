# Simplex Entropy Geometry

Tests Nesterov's entropy prox on simplex-constrained objectives. These runs cover the L1 primal and L-infinity dual geometry path.

Outputs are written to `results/runs.csv`, `results/iterations.csv`, `results/metadata.json`, and `results/summary.csv`.

## How to run

From the repository root:

```bash
python experiments/simplex_entropy_geometry/code/run.py
```

The run writes raw CSV files and creates `results/summary.csv` automatically.

Useful options:

```bash
python experiments/simplex_entropy_geometry/code/run.py --dry-run
python experiments/simplex_entropy_geometry/code/run.py --limit-runs 10
python experiments/simplex_entropy_geometry/code/run.py --output-dir /tmp/simplex_results
```

## How to change parameters

Edit `code/config.py`. The experiment matrix is built in `build_configs()`.

Important fields:

- `objective_id`: simplex objectives only.
- `D`: entropy-radius values, including `log(n)` for the full simplex.
- `prox_fun`: fixed to `entropy`.
- `dual_averaging`: `simple` or `weighted`.
- `gamma_mult`, `restrict_to_fd`, `max_iter`, `eps`: SDA controls.

Do not use entropy prox for ordinary logistic-regression weights; it is intended for probability-simplex variables.
