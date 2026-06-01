# Runtime Scaling

Generates synthetic binary logistic datasets and measures runtime and quality as sample count, dimension, and label noise vary.

Outputs are written to `results/runs.csv`, `results/iterations.csv`, `results/metadata.json`, and `results/summary.csv`.

## How to run

From the repository root:

```bash
python experiments/runtime_scaling/code/run.py
```

The run generates temporary datasets under `results/generated_data/`, writes raw CSV files, and creates `results/summary.csv` automatically.

Useful options:

```bash
python experiments/runtime_scaling/code/run.py --dry-run
python experiments/runtime_scaling/code/run.py --limit-runs 10
python experiments/runtime_scaling/code/run.py --output-dir /tmp/scaling_results
```

## How to change parameters

Edit `code/config.py`. The experiment matrix is built in `build_configs()`.

Important fields:

- `n_samples`: generated dataset size.
- `dimension`: generated feature dimension.
- `seed`: data-generation and split seed.
- `flip_prob`: independent label-flip probability.
- `D`, `gamma_mult`, `alpha`, `max_iter`, `eps`: solver controls.
- `method`: SDA variants, projected subgradient, or sklearn.

Generated CSV data is reproducible from the config and metadata.
