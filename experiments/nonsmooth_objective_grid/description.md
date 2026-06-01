# Nonsmooth Objective Grid

Broad report experiment comparing simple dual averaging, weighted dual averaging, and projected subgradient on nonsmooth objective presets with known optima.

Outputs are written to `results/runs.csv`, `results/iterations.csv`, `results/metadata.json`, and `results/summary.csv`.

## How to run

From the repository root:

```bash
python experiments/nonsmooth_objective_grid/code/run.py
```

The run writes raw CSV files and creates `results/summary.csv` automatically.

Useful options:

```bash
python experiments/nonsmooth_objective_grid/code/run.py --dry-run
python experiments/nonsmooth_objective_grid/code/run.py --limit-runs 10
python experiments/nonsmooth_objective_grid/code/run.py --output-dir /tmp/nonsmooth_results
```

## How to change parameters

Edit `code/config.py`. The experiment matrix is built in `build_configs()`.

Important fields:

- `objective_id`: registry objectives from `src/pda/objectives.py`.
- `D`: restriction radius parameter for `F_D`.
- `gamma_mult`: multiplier for SDA's theoretical `gamma*`.
- `alpha`: projected subgradient step numerator.
- `restrict_to_fd`: whether iterates are projected/mapped into `F_D`.
- `max_iter` and `eps`: iteration budget and SDA stopping tolerance.

Add/remove methods by editing the method blocks in `build_configs()`.
