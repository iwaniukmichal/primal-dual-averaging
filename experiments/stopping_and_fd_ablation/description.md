# Stopping And F_D Ablation

Isolates the effects of the restriction radius `D`, stopping tolerance `eps`, and projection onto `F_D` for simple and weighted SDA.

Outputs are written to `results/runs.csv`, `results/iterations.csv`, `results/metadata.json`, and `results/summary.csv`.

## How to run

From the repository root:

```bash
python experiments/stopping_and_fd_ablation/code/run.py
```

The run writes raw CSV files and creates `results/summary.csv` automatically.

Useful options:

```bash
python experiments/stopping_and_fd_ablation/code/run.py --dry-run
python experiments/stopping_and_fd_ablation/code/run.py --limit-runs 10
python experiments/stopping_and_fd_ablation/code/run.py --output-dir /tmp/stopping_results
```

## How to change parameters

Edit `code/config.py`. The experiment matrix is built in `build_configs()`.

Important fields:

- `objective_id`: nonsmooth registry objective rows.
- `dataset`: logistic-regression rows.
- `runner_kind`: `registry` or `logistic` for mixed objective types.
- `D`: restriction radius.
- `eps`: stopping tolerance; `0.0` disables early stopping except exact zero gap.
- `restrict_to_fd`: whether to map iterates into `F_D`.
- `gamma_mult` and `max_iter`: SDA scale and iteration budget.

Use this experiment to decide whether projected/restricted SDA behaves better for a chosen objective family.

The logistic-regression ablation uses `synthetic_logistic_small_5d.csv` so this
experiment no longer depends on the removed old demo dataset.
