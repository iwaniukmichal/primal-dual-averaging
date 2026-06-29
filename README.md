# Primal-Dual Subgradient and Dual Averaging

Python implementation and experimental study of primal-dual subgradient methods
for convex nonsmooth optimization, focused on Nesterov's dual averaging
framework.

This repo combines algorithm implementation, reproducible experiments, and a
short research report. The full write-up is available here:
[report/report.pdf](report/report.pdf).

## Project Highlights

- Implemented Simple Dual Averaging (SDA), Weighted Dual Averaging (WDA),
  Stochastic Simple Averages (SSA), and projected subgradient descent.
- Added Euclidean, weighted Euclidean, and entropy prox geometries.
- Built benchmark nonsmooth convex objectives with known optima.
- Built lasso logistic regression experiments with synthetic and public binary
  classification datasets.
- Compared custom solvers against `sklearn` SAGA for lasso logistic regression.
- Added experiment runners, CSV result exports, plotting scripts, and unit tests.

## Research Summary

The project investigates how primal-dual subgradient and dual averaging methods
behave in practice, especially when the objective is convex but nonsmooth.

The main takeaways from the report are:

- The primal-dual certificate is useful, but only when the prox-radius parameter
  `D` is chosen large enough to contain an optimum.
- `D` and `gamma_mult` are important tuning parameters and should be chosen
  together.
- Weighted Dual Averaging performed best among the custom methods on the lasso
  logistic regression grid.
- Projected subgradient descent remained a strong baseline on several small
  nonsmooth benchmark problems.
- Weighted prox geometry helps when it matches coordinate scaling, but can hurt
  when the geometry is mismatched.
- SSA reduces per-iteration cost, but in the tested setup it did not match the
  final objective gaps reached by full-gradient WDA.

## Repository Layout

```text
src/pda/       Solver and objective implementations
experiments/   Experiment configs, runners, plotting scripts, saved results
data/          Dataset generation and download scripts
report/        LaTeX report, compiled PDF, and figures
tests/         Unit and smoke tests
docs/          Supporting notes and derivations
```

## Setup

The repo is plain Python and currently has no package metadata file. From the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly pytest
export PYTHONPATH="$PWD/src:$PWD"
```

## Quick Usage

Run the test suite:

```bash
PYTHONPATH=src python -m pytest
```

Run a small experiment smoke test:

```bash
python experiments/run_all.py --experiment nonsmooth_objective_grid --limit-runs 5
```

List all experiment groups:

```bash
python experiments/run_all.py --list
```

Run all experiments:

```bash
python experiments/run_all.py
```

Generate report figures from existing experiment results:

```bash
python report/make_report_plots.py
```

Regenerate public binary datasets:

```bash
python data/download_binary_datasets.py
```

## Example Solver Call

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

print(result["x_hat"][-1])
print(result["gap"][-1])
```

## Outputs

Experiment runs write results under:

```text
experiments/<experiment_name>/results/
```

Each result directory contains `runs.csv`, `iterations.csv`, `summary.csv`, and
`metadata.json`.

## Report

Read the report: [report/report.pdf](report/report.pdf)

Rebuild it with LaTeX:

```bash
cd report
latexmk -pdf report.tex
```
