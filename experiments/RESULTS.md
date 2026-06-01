# Experiment Results Reference

Each experiment writes its outputs to its local `results/` directory by default:

```text
experiments/<experiment_name>/results/
  runs.csv
  iterations.csv
  metadata.json
  summary.csv
```

Use `--output-dir <path>` to write the same files somewhere else.

## Files

`runs.csv`

One row per executed method/configuration. Use this file for final comparison tables.

`iterations.csv`

One row per iteration for iterative methods. Use this file for convergence plots.
Sklearn rows do not appear here because sklearn does not expose matching per-iteration
trajectory metrics through this experiment runner.

`metadata.json`

Run metadata for reproducibility: experiment name, timestamp, Python/platform info,
git commit when available, configured/selected/completed run counts, skipped configs,
and the full parameter grid.

`summary.csv`

Compact aggregate table generated automatically by `run.py` after the raw files are
written. It groups rows by method and objective/dataset.

## `runs.csv` Columns

| Column | Meaning |
| --- | --- |
| `experiment` | Experiment folder/name that produced the row. |
| `run_id` | Stable identifier derived from the experiment name and full config. |
| `method` | Method label from `code/config.py`, e.g. `sda_simple_euclidean`, `projected_subgradient`, `sklearn_saga`. |
| `solver_family` | Broader implementation family, e.g. simple/weighted dual averaging, projected subgradient, sklearn logistic regression. |
| `objective_id` | Registry objective ID or constructed logistic objective ID. |
| `dataset` | Dataset path for logistic-regression experiments; blank for registry objective experiments. |
| `objective_family` | Objective category, e.g. `max_affine`, `logistic_regression`, `lasso_logistic_regression`. |
| `dimension` | Optimization variable dimension. Logistic regression includes the intercept coordinate. |
| `prox_fun` | Prox-function used by SDA, e.g. `euclidean`, `weighted_euclidean`, `entropy`; blank for methods without SDA prox. |
| `prox_weights_profile` | Weight profile name for weighted Euclidean prox; blank when unused. |
| `dual_averaging` | SDA averaging mode: `simple` or `weighted`; blank for non-SDA methods. |
| `D` | Restriction parameter for `F_D = {x : d(x) <= D}`. |
| `gamma` | Actual SDA gamma used after applying `gamma_mult`; blank for non-SDA methods. |
| `gamma_mult` | Multiplier applied to theoretical `gamma*`. |
| `alpha` | Projected subgradient numerator used in `alpha_k = alpha / sqrt(k + 1)`. |
| `lambda` | L1 penalty coefficient for lasso logistic regression. |
| `eps` | SDA stopping tolerance for normalized gap. |
| `max_iter` | Maximum iterations for the method. |
| `restrict_to_fd` | Whether iterates are mapped/projected into `F_D`. |
| `seed` | Dataset split or generation seed, when applicable. |
| `iterations` | Number of iterations actually run. |
| `converged` | Method convergence flag. For subgradient this is currently always false. |
| `runtime_seconds` | Total measured runtime for the method call. |
| `avg_iteration_time_seconds` | Runtime divided by iterations. |
| `objective_gap` | Final objective value gap against known optimum for registry objectives. |
| `train_loss` | Final train loss for logistic-regression objectives. |
| `test_loss` | Final test loss for logistic-regression objectives. |
| `test_accuracy` | Final test accuracy for logistic-regression objectives. |
| `nonzero_count` | Number of nonzero final parameters, mainly useful for lasso. |
| `final_norm` | Euclidean norm of the final averaged iterate or final sklearn parameter vector. |
| `final_gap` | Final normalized SDA gap when available; otherwise objective gap where relevant. |
| `final_x` | JSON-encoded final raw iterate for iterative methods. |
| `final_x_hat` | JSON-encoded final averaged iterate for iterative methods. |
| `final_parameter_vector` | JSON-encoded sklearn parameter vector for sklearn logistic runs. |

Blank cells mean the field is not applicable to that method/objective.

## `iterations.csv` Columns

| Column | Meaning |
| --- | --- |
| `run_id` | Links the row back to `runs.csv`. |
| `iteration` | 1-based iteration number. |
| `elapsed_estimate_seconds` | Estimated elapsed time using `iteration * avg_iteration_time_seconds`. |
| `objective_value_x` | Objective or train loss evaluated at raw iterate `x_k`. |
| `objective_value_x_hat` | Objective or train loss evaluated at averaged iterate `x_hat_k`. |
| `objective_gap_x` | Objective gap at `x_k` for objectives with known optimum. |
| `objective_gap_x_hat` | Objective gap at `x_hat_k` for objectives with known optimum. |
| `normalized_gap` | SDA normalized gap. Blank for projected subgradient. |
| `x_norm` | Euclidean norm of raw iterate. |
| `x_hat_norm` | Euclidean norm of averaged iterate. |
| `g_norm` | Euclidean norm of the subgradient used at that iteration. |
| `nonzero_count` | Number of nonzero averaged parameters when available. |

## `summary.csv` Columns

| Column | Meaning |
| --- | --- |
| `method` | Method label. |
| `objective_id` | Objective ID for registry or logistic objective rows. |
| `dataset` | Dataset path for logistic rows. |
| `runs` | Number of rows aggregated into this summary row. |
| `mean_runtime_seconds` | Mean `runtime_seconds` over the group. |
| `mean_final_gap` | Mean final gap over rows where a final gap is defined. |
| `mean_test_accuracy` | Mean test accuracy over logistic rows. |

## Notes

- Vector-valued cells are JSON strings so they can be parsed reproducibly.
- Full vector trajectories are intentionally not stored; only scalar trajectories go
  into `iterations.csv`.
- To change an experiment matrix, edit that experiment's `code/config.py`.
- To run a small smoke test, use `--limit-runs`, for example:

```bash
python experiments/nonsmooth_objective_grid/code/run.py --limit-runs 5
```

