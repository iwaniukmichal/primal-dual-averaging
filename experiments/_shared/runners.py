from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .csv_io import ITERATION_COLUMNS, RUN_COLUMNS, read_csv, write_csv
from .metrics import gamma_star, to_jsonable, trajectory_metric, value_norm
from .objectives import (
    build_logreg_objective,
    build_registry_objective,
    data_path,
    generate_temporary_logistic_dataset,
    prox_center_for_objective,
    prox_weights_for_profile,
)
from pda import SDA, SubgradientMethod


def stable_run_id(experiment: str, config: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(config), sort_keys=True, separators=(",", ":"))
    return f"{experiment}:{payload}"


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: ../results relative to the run.py file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print run count without writing outputs.")
    parser.add_argument("--limit-runs", type=int, default=None, help="Execute only the first N runs.")
    return parser


def default_results_dir(run_file: str | Path) -> Path:
    return Path(run_file).resolve().parents[1] / "results"


def run_registry_method(config: dict[str, Any], experiment: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    objective = build_registry_objective(str(config["objective_id"]))
    method = str(config["method"])
    run_id = stable_run_id(experiment, config)

    if method.startswith("sda_"):
        return _run_registry_sda(objective, config, experiment, run_id)
    if method == "projected_subgradient":
        return _run_registry_subgradient(objective, config, experiment, run_id)
    raise ValueError(f"Unsupported registry method: {method}")


def run_logistic_method(config: dict[str, Any], experiment: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = Path(config["dataset"])
    if not dataset.is_absolute():
        dataset = data_path(str(dataset))
    objective = build_logreg_objective(
        dataset,
        lasso=bool(config.get("lasso", False)),
        lasso_lambda=float(config.get("lambda", 1.0)),
        test_size=float(config.get("test_size", 0.2)),
        seed=int(config.get("seed", 0)),
    )
    method = str(config["method"])
    run_id = stable_run_id(experiment, {**config, "dataset": str(dataset)})

    if method.startswith("sda_"):
        return _run_logistic_sda(objective, config, experiment, run_id, dataset)
    if method == "ssa_euclidean":
        return _run_logistic_ssa(objective, config, experiment, run_id, dataset)
    if method == "projected_subgradient":
        return _run_logistic_subgradient(objective, config, experiment, run_id, dataset)
    if method == "sklearn_saga":
        return _run_sklearn_logistic(objective, config, experiment, run_id, dataset)
    raise ValueError(f"Unsupported logistic method: {method}")


def run_generated_logistic_method(
    config: dict[str, Any],
    experiment: str,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = generate_temporary_logistic_dataset(
        output_dir,
        n_samples=int(config["n_samples"]),
        dimension=int(config["dimension"]),
        seed=int(config["seed"]),
        flip_prob=float(config["flip_prob"]),
    )
    return run_logistic_method({**config, "dataset": str(dataset)}, experiment)


def execute_experiment(
    *,
    experiment: str,
    configs: list[dict[str, Any]],
    output_dir: Path,
    runner_kind: str,
    dry_run: bool,
    limit_runs: int | None,
) -> None:
    selected_configs = configs[: max(0, limit_runs)] if limit_runs is not None else configs
    if dry_run:
        print(f"{experiment}: {len(configs)} configured runs ({len(selected_configs)} selected)")
        return

    run_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    progress_started_at = perf_counter()
    _print_progress(
        experiment=experiment,
        processed=0,
        total=len(selected_configs),
        completed=0,
        skipped=0,
        started_at=progress_started_at,
    )

    for index, config in enumerate(selected_configs, start=1):
        try:
            effective_runner_kind = str(config.get("runner_kind", runner_kind))
            if effective_runner_kind == "registry":
                run_row, iter_rows = run_registry_method(config, experiment)
            elif effective_runner_kind == "logistic":
                run_row, iter_rows = run_logistic_method(config, experiment)
            elif effective_runner_kind == "generated_logistic":
                run_row, iter_rows = run_generated_logistic_method(config, experiment, output_dir)
            else:
                raise ValueError(f"Unknown runner kind: {effective_runner_kind}")
        except Exception as exc:
            skipped.append({"index": index, "config": to_jsonable(config), "reason": str(exc)})
            _print_progress(
                experiment=experiment,
                processed=index,
                total=len(selected_configs),
                completed=len(run_rows),
                skipped=len(skipped),
                started_at=progress_started_at,
            )
            continue
        run_rows.append(run_row)
        iteration_rows.extend(iter_rows)
        _print_progress(
            experiment=experiment,
            processed=index,
            total=len(selected_configs),
            completed=len(run_rows),
            skipped=len(skipped),
            started_at=progress_started_at,
        )

    if not selected_configs:
        _print_progress(
            experiment=experiment,
            processed=0,
            total=0,
            completed=0,
            skipped=0,
            started_at=progress_started_at,
        )
    print(file=sys.stderr)

    write_csv(output_dir / "runs.csv", run_rows, RUN_COLUMNS)
    write_csv(output_dir / "iterations.csv", iteration_rows, ITERATION_COLUMNS)
    metadata = {
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "configured_run_count": len(configs),
        "selected_run_count": len(selected_configs),
        "completed_run_count": len(run_rows),
        "skipped": skipped,
        "grid": to_jsonable(configs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(run_rows)} runs to {output_dir / 'runs.csv'}")
    if skipped:
        print(f"Skipped {len(skipped)} invalid runs; see metadata.json")


def _print_progress(
    *,
    experiment: str,
    processed: int,
    total: int,
    completed: int,
    skipped: int,
    started_at: float,
) -> None:
    if total <= 0:
        message = f"{experiment}: no selected configs"
    else:
        width = 30
        filled = int(width * processed / total)
        bar = "#" * filled + "-" * (width - filled)
        percent = 100.0 * processed / total
        elapsed = perf_counter() - started_at
        message = (
            f"{experiment}: [{bar}] {processed}/{total} configs "
            f"({percent:5.1f}%) completed={completed} skipped={skipped} "
            f"elapsed={elapsed:.1f}s"
        )
    print(f"\r{message}", end="", file=sys.stderr, flush=True)


def analyze_results(results_dir: Path) -> None:
    runs_path = results_dir / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing runs.csv: {runs_path}")
    rows = read_csv(runs_path)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("method", ""), row.get("objective_id", ""), row.get("dataset", ""))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method, objective_id, dataset), group_rows in sorted(groups.items()):
        runtimes = [_as_float(row.get("runtime_seconds")) for row in group_rows]
        gaps = [_as_float(row.get("final_gap") or row.get("objective_gap")) for row in group_rows]
        accuracies = [_as_float(row.get("test_accuracy")) for row in group_rows]
        summary_rows.append(
            {
                "method": method,
                "objective_id": objective_id,
                "dataset": dataset,
                "runs": len(group_rows),
                "mean_runtime_seconds": _mean_defined(runtimes),
                "mean_final_gap": _mean_defined(gaps),
                "mean_test_accuracy": _mean_defined(accuracies),
            }
        )
    write_csv(
        results_dir / "summary.csv",
        summary_rows,
        [
            "method",
            "objective_id",
            "dataset",
            "runs",
            "mean_runtime_seconds",
            "mean_final_gap",
            "mean_test_accuracy",
        ],
    )
    print(f"Wrote {results_dir / 'summary.csv'}")


def _run_registry_sda(
    objective: Any,
    config: dict[str, Any],
    experiment: str,
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prox_fun = str(config.get("prox_fun", "euclidean"))
    dual_averaging = str(config.get("dual_averaging", "simple"))
    profile = str(config.get("prox_weights_profile", ""))
    prox_center = prox_center_for_objective(int(objective.dimension), prox_fun)
    prox_weights = prox_weights_for_profile(int(objective.dimension), profile)
    gamma = float(config.get("gamma_mult", 1.0)) * gamma_star(
        float(config["D"]),
        float(objective.lipschitz_constant),
    )
    result = SDA(
        prox_center=prox_center,
        prox_fun=prox_fun,
        prox_weights=prox_weights,
    ).run(
        gamma=gamma,
        D=float(config["D"]),
        eps=float(config.get("eps", 1e-4)),
        subgradient_oracle=objective.subgradient,
        max_iter=int(config.get("max_iter", 1000)),
        restrict_to_fd=bool(config.get("restrict_to_fd", False)),
        dual_averaging=dual_averaging,
    )
    return _summarize_registry_result(experiment, run_id, config, objective, result, gamma)


def _run_registry_subgradient(
    objective: Any,
    config: dict[str, Any],
    experiment: str,
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prox_center = prox_center_for_objective(int(objective.dimension), "euclidean")
    result = SubgradientMethod(prox_center=prox_center).run(
        gamma=float(config["alpha"]),
        D=float(config["D"]),
        subgradient_oracle=objective.subgradient,
        max_iter=int(config.get("max_iter", 1000)),
        restrict_to_fd=bool(config.get("restrict_to_fd", False)),
    )
    return _summarize_registry_result(experiment, run_id, config, objective, result, None)


def _summarize_registry_result(
    experiment: str,
    run_id: str,
    config: dict[str, Any],
    objective: Any,
    result: dict[str, Any],
    gamma: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    iterations = int(result["iterations"])
    x_values = list(result["x"][:iterations])
    x_hat_values = list(result["x_hat"][1 : iterations + 1])
    f_x = trajectory_metric(objective.objective, x_values)
    f_x_hat = trajectory_metric(objective.objective, x_hat_values)
    objective_gap_x = [value - float(objective.minimum_value) for value in f_x]
    objective_gap_x_hat = [value - float(objective.minimum_value) for value in f_x_hat]
    normalized_gap = _normalized_gap(result)

    final_x = result["x"][-1]
    final_x_hat = result["x_hat"][-1]
    final_objective = float(objective.objective(final_x_hat))
    final_gap = final_objective - float(objective.minimum_value)
    run_row = _base_run_row(experiment, run_id, config, result)
    run_row.update(
        {
            "solver_family": result.get("dual_averaging", config.get("method")),
            "objective_id": objective.id,
            "objective_family": objective.family,
            "dimension": objective.dimension,
            "gamma": gamma,
            "objective_gap": final_gap,
            "final_gap": normalized_gap[-1] if normalized_gap else final_gap,
            "final_norm": value_norm(final_x_hat),
            "final_x": to_jsonable(final_x),
            "final_x_hat": to_jsonable(final_x_hat),
        }
    )
    iteration_rows = _iteration_rows(
        run_id,
        result,
        objective_values_x=f_x,
        objective_values_x_hat=f_x_hat,
        objective_gaps_x=objective_gap_x,
        objective_gaps_x_hat=objective_gap_x_hat,
        normalized_gap=normalized_gap,
    )
    return run_row, iteration_rows


def _run_logistic_sda(
    objective: Any,
    config: dict[str, Any],
    experiment: str,
    run_id: str,
    dataset: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gamma = float(config.get("gamma_mult", 1.0)) * gamma_star(
        float(config["D"]),
        float(objective.lipschitz_constant),
    )
    prox_center = prox_center_for_objective(int(objective.dimension), "euclidean")
    result = SDA(prox_center=prox_center).run(
        gamma=gamma,
        D=float(config["D"]),
        eps=float(config.get("eps", 1e-4)),
        subgradient_oracle=objective.subgradient,
        max_iter=int(config.get("max_iter", 1000)),
        restrict_to_fd=bool(config.get("restrict_to_fd", False)),
        dual_averaging=str(config.get("dual_averaging", "simple")),
    )
    return _summarize_logistic_result(experiment, run_id, config, objective, result, gamma, dataset)


def _run_logistic_ssa(
    objective: Any,
    config: dict[str, Any],
    experiment: str,
    run_id: str,
    dataset: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lasso_lambda = float(config.get("lambda", 1.0)) if bool(config.get("lasso", False)) else 0.0
    sample_lipschitz = float(np.max(np.linalg.norm(objective.X_train, axis=1)))
    if lasso_lambda > 0.0:
        train_sample_count = int(objective.X_train.shape[0])
        sample_lipschitz += float((lasso_lambda / train_sample_count) * np.sqrt(objective.dimension))
    gamma = float(config.get("gamma_mult", 1.0)) * gamma_star(
        float(config["D"]),
        sample_lipschitz,
    )
    rng = np.random.default_rng(int(config.get("sample_seed", config.get("seed", 0))))
    subgradient_oracle = _logistic_stochastic_subgradient_oracle(
        objective,
        lasso_lambda=lasso_lambda,
        batch_size=int(config.get("batch_size", 1)),
        rng=rng,
    )
    prox_center = prox_center_for_objective(int(objective.dimension), "euclidean")
    result = SDA(prox_center=prox_center).run(
        gamma=gamma,
        D=float(config["D"]),
        eps=float(config.get("eps", 1e-4)),
        subgradient_oracle=subgradient_oracle,
        max_iter=int(config.get("max_iter", 1000)),
        restrict_to_fd=bool(config.get("restrict_to_fd", False)),
        dual_averaging="simple",
        stop_on_gap=False,
    )
    result["dual_averaging"] = "stochastic_simple"
    return _summarize_logistic_result(experiment, run_id, config, objective, result, gamma, dataset)


def _logistic_stochastic_subgradient_oracle(
    objective: Any,
    *,
    lasso_lambda: float,
    batch_size: int,
    rng: np.random.Generator,
) -> Any:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for SSA.")

    X_train = np.asarray(objective.X_train, dtype=float)
    y_train = np.asarray(objective.y_train, dtype=float)
    train_sample_count = int(X_train.shape[0])
    dimension = int(objective.dimension)

    def oracle(weights: Any) -> np.ndarray:
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape != (dimension,):
            raise ValueError(
                f"Expected a vector of shape {(dimension,)}, got {weights_array.shape}."
            )
        indices = rng.integers(0, train_sample_count, size=batch_size)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        logits = np.nan_to_num(X_batch @ weights_array, nan=0.0, posinf=500.0, neginf=-500.0)
        positive = logits >= 0.0
        probabilities = np.empty_like(logits, dtype=float)
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_values = np.exp(logits[~positive])
        probabilities[~positive] = exp_values / (1.0 + exp_values)
        gradient = (X_batch.T @ (probabilities - y_batch)) / batch_size
        if lasso_lambda > 0.0:
            penalty_gradient = np.zeros_like(weights_array)
            penalty_signs = np.sign(weights_array[:-1])
            penalty_signs[np.isclose(weights_array[:-1], 0.0)] = 0.0
            penalty_gradient[:-1] = penalty_signs
            gradient = gradient + (lasso_lambda / train_sample_count) * penalty_gradient
        return gradient

    return oracle


def _run_logistic_subgradient(
    objective: Any,
    config: dict[str, Any],
    experiment: str,
    run_id: str,
    dataset: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prox_center = prox_center_for_objective(int(objective.dimension), "euclidean")
    result = SubgradientMethod(prox_center=prox_center).run(
        gamma=float(config["alpha"]),
        D=float(config["D"]),
        subgradient_oracle=objective.subgradient,
        max_iter=int(config.get("max_iter", 1000)),
        restrict_to_fd=bool(config.get("restrict_to_fd", False)),
    )
    return _summarize_logistic_result(experiment, run_id, config, objective, result, None, dataset)


def _run_sklearn_logistic(
    objective: Any,
    config: dict[str, Any],
    experiment: str,
    run_id: str,
    dataset: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    import warnings

    penalty = None
    kwargs: dict[str, Any] = {
        "solver": "saga",
        "penalty": None,
        "fit_intercept": True,
        "max_iter": int(config.get("max_iter", 1000)),
        "random_state": int(config.get("seed", 0)),
    }
    if bool(config.get("lasso", False)):
        penalty = "l1"
        lasso_lambda = max(float(config.get("lambda", 1.0)), 1e-12)
        kwargs.update({"penalty": "l1", "C": 1.0 / lasso_lambda})

    model = LogisticRegression(**kwargs)
    X_train_raw = objective.X_train[:, :-1]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", ConvergenceWarning)
        start = perf_counter()
        model.fit(X_train_raw, objective.y_train.astype(int))
        runtime = perf_counter() - start

    parameter_vector = np.concatenate(
        [np.asarray(model.coef_[0], dtype=float), np.asarray([float(model.intercept_[0])])]
    )
    iterations = int(np.max(np.asarray(model.n_iter_, dtype=int)))
    run_row = _base_run_row(
        experiment,
        run_id,
        {**config, "method": "sklearn_saga"},
        {
            "iterations": iterations,
            "converged": iterations < int(config.get("max_iter", 1000)),
            "total_runtime_seconds": runtime,
            "avg_iteration_time_seconds": runtime / iterations if iterations else 0.0,
            "restrict_to_fd": False,
        },
    )
    run_row.update(
        {
            "solver_family": "sklearn_logistic_regression",
            "objective_id": objective.id,
            "dataset": str(dataset),
            "objective_family": objective.family,
            "dimension": objective.dimension,
            "train_loss": float(objective.train_loss(parameter_vector)),
            "test_loss": float(objective.test_loss(parameter_vector)),
            "test_accuracy": float(objective.test_accuracy(parameter_vector)),
            "nonzero_count": int(objective.nonzero_count(parameter_vector)),
            "final_norm": value_norm(parameter_vector),
            "final_parameter_vector": to_jsonable(parameter_vector),
            "lambda": config.get("lambda"),
            "prox_fun": "",
            "dual_averaging": "",
        }
    )
    return run_row, []


def _summarize_logistic_result(
    experiment: str,
    run_id: str,
    config: dict[str, Any],
    objective: Any,
    result: dict[str, Any],
    gamma: float | None,
    dataset: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    iterations = int(result["iterations"])
    x_values = list(result["x"][:iterations])
    x_hat_values = list(result["x_hat"][1 : iterations + 1])
    train_x = trajectory_metric(objective.train_loss, x_values)
    train_x_hat = trajectory_metric(objective.train_loss, x_hat_values)
    test_x_hat = trajectory_metric(objective.test_loss, x_hat_values)
    accuracy_x_hat = trajectory_metric(objective.test_accuracy, x_hat_values)
    nonzero_x_hat = trajectory_metric(objective.nonzero_count, x_hat_values, cast=int)
    normalized_gap = _normalized_gap(result)

    final_x_hat = result["x_hat"][-1]
    run_row = _base_run_row(experiment, run_id, config, result)
    run_row.update(
        {
            "solver_family": result.get("dual_averaging", config.get("method")),
            "objective_id": objective.id,
            "dataset": str(dataset),
            "objective_family": objective.family,
            "dimension": objective.dimension,
            "gamma": gamma,
            "train_loss": float(objective.train_loss(final_x_hat)),
            "test_loss": float(objective.test_loss(final_x_hat)),
            "test_accuracy": float(objective.test_accuracy(final_x_hat)),
            "nonzero_count": int(objective.nonzero_count(final_x_hat)),
            "final_norm": value_norm(final_x_hat),
            "final_gap": normalized_gap[-1] if normalized_gap else None,
            "final_x": to_jsonable(result["x"][-1]),
            "final_x_hat": to_jsonable(final_x_hat),
        }
    )
    iteration_rows = _iteration_rows(
        run_id,
        result,
        objective_values_x=train_x,
        objective_values_x_hat=train_x_hat,
        objective_gaps_x=[None] * len(train_x),
        objective_gaps_x_hat=[None] * len(train_x_hat),
        normalized_gap=normalized_gap,
        nonzero_count=nonzero_x_hat,
    )
    for row, test_loss, accuracy in zip(iteration_rows, test_x_hat, accuracy_x_hat):
        row["test_loss"] = test_loss
        row["test_accuracy"] = accuracy
    return run_row, iteration_rows


def _base_run_row(
    experiment: str,
    run_id: str,
    config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "run_id": run_id,
        "method": config.get("method"),
        "prox_fun": result.get("prox_fun", config.get("prox_fun", "")),
        "prox_weights_profile": config.get("prox_weights_profile", ""),
        "dual_averaging": result.get("dual_averaging", config.get("dual_averaging", "")),
        "D": config.get("D"),
        "gamma_mult": config.get("gamma_mult"),
        "alpha": config.get("alpha"),
        "lambda": config.get("lambda"),
        "eps": config.get("eps"),
        "max_iter": config.get("max_iter"),
        "restrict_to_fd": result.get("restrict_to_fd", config.get("restrict_to_fd")),
        "seed": config.get("seed"),
        "iterations": result.get("iterations"),
        "converged": result.get("converged", False),
        "runtime_seconds": result.get("total_runtime_seconds"),
        "avg_iteration_time_seconds": result.get("avg_iteration_time_seconds"),
    }


def _iteration_rows(
    run_id: str,
    result: dict[str, Any],
    *,
    objective_values_x: list[Any],
    objective_values_x_hat: list[Any],
    objective_gaps_x: list[Any],
    objective_gaps_x_hat: list[Any],
    normalized_gap: list[Any],
    nonzero_count: list[Any] | None = None,
) -> list[dict[str, Any]]:
    iterations = int(result["iterations"])
    avg_time = float(result.get("avg_iteration_time_seconds", 0.0))
    g_values = list(result.get("g", []))
    rows: list[dict[str, Any]] = []
    for index in range(iterations):
        rows.append(
            {
                "run_id": run_id,
                "iteration": index + 1,
                "elapsed_estimate_seconds": (index + 1) * avg_time,
                "objective_value_x": objective_values_x[index] if index < len(objective_values_x) else None,
                "objective_value_x_hat": (
                    objective_values_x_hat[index] if index < len(objective_values_x_hat) else None
                ),
                "objective_gap_x": objective_gaps_x[index] if index < len(objective_gaps_x) else None,
                "objective_gap_x_hat": (
                    objective_gaps_x_hat[index] if index < len(objective_gaps_x_hat) else None
                ),
                "normalized_gap": normalized_gap[index] if index < len(normalized_gap) else None,
                "x_norm": value_norm(result["x"][index]),
                "x_hat_norm": value_norm(result["x_hat"][index + 1]),
                "g_norm": value_norm(g_values[index]) if index < len(g_values) else None,
                "nonzero_count": nonzero_count[index] if nonzero_count and index < len(nonzero_count) else None,
            }
        )
    return rows


def _normalized_gap(result: dict[str, Any]) -> list[float]:
    gaps = list(result.get("gap", []))
    if not gaps:
        return []
    S = result.get("S")
    if S:
        return [float(gap) / (float(S[index + 1]) or 1.0) for index, gap in enumerate(gaps)]
    return [float(gap) / (index + 1) for index, gap in enumerate(gaps)]


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None and np.isfinite(value)]
    if not defined:
        return None
    return float(np.mean(defined))
