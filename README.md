# Primal-Dual Subgradient and Dual Averaging

Python implementation and experimental study of
primal-dual subgradient methods for convex nonsmooth optimization, with a focus
on Nesterov's dual averaging framework. The full write-up is available in
[report/report.pdf](report/report.pdf).

The code implements deterministic and stochastic dual averaging methods,
benchmark nonsmooth objectives with known optima, lasso logistic regression
objectives, experiment runners, plotting scripts, and tests.

## What Is Implemented

- Simple Dual Averaging (SDA) with Euclidean, weighted Euclidean, and entropy
  prox geometries.
- Weighted Dual Averaging (WDA), where each nonzero subgradient is normalized by
  its dual norm before being accumulated.
- Stochastic Simple Averages (SSA) for finite-sum lasso logistic regression,
  using sampled training examples.
- Projected subgradient descent as a Euclidean baseline.
- Deterministic nonsmooth objective registry, including absolute-value,
  max-affine, shifted `l_inf`, weighted `l1`, ill-conditioned, and
  simplex-constrained objectives.
- Dataset-backed lasso logistic regression objectives with train/test splits,
  feature standardization, unpenalized intercepts, and `sklearn` SAGA reference
  runs.
- Experiment infrastructure that writes `runs.csv`, `iterations.csv`,
  `summary.csv`, and `metadata.json`.
- Plotting scripts for interactive experiment plots and report-ready static
  figures.

Core solver code lives in [src/pda](src/pda). Experiment definitions and result
pipelines live in [experiments](experiments).

## Research Summary

The report studies how primal-dual subgradient methods behave on nonsmooth
convex objectives and lasso logistic regression problems. The main research
questions are:

- when the primal-dual certificate gap is a valid stopping certificate;
- how the prox-radius parameter `D` and step multiplier `gamma_mult` affect
  convergence;
- whether weighted dual averaging improves over simple dual averaging and
  projected subgradient descent;
- when weighted Euclidean or entropy prox geometry helps;
- whether stochastic sampled subgradients can trade accuracy for lower runtime.

The main findings from [report/report.pdf](report/report.pdf) are:

- The primal-dual certificate gap is valid only when the chosen prox radius
  satisfies `D >= d(x*)`. If `D` is too small, the certificate can become
  invalid or negative and may stop a run while the true objective gap is still
  large.
- Even with a valid radius, the certificate can be more conservative than the
  direct objective gap. The experiments therefore use objective gaps for method
  comparison and treat the certificate as a diagnostic.
- On the nonsmooth benchmark grid, projected subgradient descent has the best
  median final objective gap, while WDA achieves the best single tuned run.
- On lasso logistic regression datasets, WDA is the strongest custom method in
  the stored experiment grid, reaching the smallest median train-objective gap
  to the matched `sklearn` SAGA reference.
- `D` and `gamma_mult` must be tuned together. The theoretical baseline is a
  useful reference point, but it is not uniformly optimal empirically.
- Weighted Euclidean geometry helps only when the diagonal weights match the
  coordinate structure of the objective; mismatched geometry can make results
  worse.
- Entropy prox is appropriate for simplex-constrained objectives, but simple
  entropy SDA and weighted entropy WDA have similar median final gaps in the
  tested grid.
- SSA has cheaper iterations than full-gradient WDA, but within the tested
  1000-iteration budget it leaves larger train-objective gaps on synthetic
  lasso datasets.

## Repository Layout

```text
src/pda/                       Solver and objective implementations
data/                          Dataset generation/download scripts
experiments/                   Experiment grids, runners, plotting scripts
report/                        LaTeX report, PDF, and report figures
tests/                         Unit and smoke tests
docs/                          Notes and supporting derivations
```

## Setup

This repository does not currently include a packaged dependency file. Create a
virtual environment and install the dependencies used by the code:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly pytest
```

Use the repository root as the working directory. For direct imports from
`src/pda`, set `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PWD"
```

The experiment entry points bootstrap this path automatically, but setting
`PYTHONPATH` is useful for tests and ad hoc scripts.

## Run Tests

```bash
PYTHONPATH=src python -m pytest
```

## Use The Solvers Directly

Example: run SDA on a built-in nonsmooth objective.

```python
from pda import SDA, get_objective

objective = get_objective("abs_a2")
solver = SDA(prox_center=0.0, prox_fun="euclidean")

result = solver.run(
    gamma=1.0,
    D=8.0,
    eps=1e-3,
    subgradient_oracle=objective.subgradient,
    max_iter=1000,
)

print(result["iterations"])
print(result["x_hat"][-1])
print(result["gap"][-1])
```

## Run Experiments

List available experiment groups:

```bash
python experiments/run_all.py --list
```

Run a small smoke test:

```bash
python experiments/run_all.py --experiment nonsmooth_objective_grid --limit-runs 5
```

Run all configured experiments:

```bash
python experiments/run_all.py
```

By default, each experiment writes results under:

```text
experiments/<experiment_name>/results/
```

Each result directory contains:

- `runs.csv`: one row per method/configuration;
- `iterations.csv`: per-iteration trajectory metrics;
- `summary.csv`: grouped summary metrics;
- `metadata.json`: reproducibility metadata.

Use a separate output root if you do not want to overwrite the checked-in result
directories:

```bash
python experiments/run_all.py --output-root /tmp/pda-results --limit-runs 10
```

## Generate Plots

Generate interactive plots from experiment results:

```bash
python experiments/plot_all.py
```

Generate report-ready PDF/PNG figures:

```bash
python report/make_report_plots.py
```

Limit report figures to one group:

```bash
python report/make_report_plots.py --plots lasso
```

## Data

Synthetic datasets can be regenerated with:

```bash
python data/generate_logistic_data.py --help
```

Public binary classification datasets are downloaded and normalized with:

```bash
python data/download_binary_datasets.py
```

More details are in [data/README.md](data/README.md).

## Report

The compiled report is [report/report.pdf](report/report.pdf). The source is
[report/report.tex](report/report.tex). If a LaTeX distribution with `latexmk`
is installed, rebuild the PDF with:

```bash
cd report
latexmk -pdf report.tex
```
