from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TypeVar

import numpy as np

from pda import SDA, SubgradientMethod  # noqa


class _HasDimension(Protocol):
    dimension: int


class _HasObjectiveMetadata(_HasDimension, Protocol):
    id: str
    family: str
    name: str
    params: Mapping[str, Any]
    lipschitz_constant: float


class _HasKnownMinimum(_HasObjectiveMetadata, Protocol):
    minimum_value: float
    minimizer: Any


class _HasSDAInputs(_HasDimension, Protocol):
    lipschitz_constant: float

    def subgradient(self, value: Any) -> Any:
        ...


SummaryResult = TypeVar("SummaryResult", bound=dict[str, Any])


def gamma_star(D: float, L: float = 1.0, sigma: float = 1.0) -> float:
    """Return the theoretical SDA parameter `gamma* = L / sqrt(2 sigma D)`."""
    return float(L / np.sqrt(2.0 * sigma * D))


def _value_norm(value: Any) -> float:
    """Return the Euclidean norm of a scalar or vector value."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(abs(array))
    return float(np.linalg.norm(array))


def _to_jsonable(value: Any) -> Any:
    """Convert NumPy-backed values to JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(inner_value) for inner_value in value]
    return value


def _normalize_run_id_value(value: Any) -> Any:
    """Convert values to stable, JSON-safe structures for run-id serialization."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_run_id_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_run_id_value(inner_value) for inner_value in value]
    return _to_jsonable(value)


def _stringify_run_id_value(value: Any) -> str:
    """Return a deterministic string representation for one run-id value."""
    normalized = _normalize_run_id_value(value)
    if isinstance(normalized, (dict, list)):
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return str(normalized)


def build_run_id(args: argparse.Namespace | Mapping[str, Any]) -> str:
    """Build a deterministic human-readable run id from parsed arguments."""
    if isinstance(args, argparse.Namespace):
        args_mapping = vars(args)
    else:
        args_mapping = dict(args)

    parts = []
    for key, value in sorted(args_mapping.items(), key=lambda item: str(item[0])):
        parts.append(f"{key}={_stringify_run_id_value(value)}")
    return "|".join(parts)


def _zero_prox_center(objective: _HasDimension) -> float | np.ndarray:
    """Return a zero prox-center matching the objective dimension."""
    if objective.dimension == 1:
        return 0.0
    return np.zeros(objective.dimension, dtype=float)


def _prepare_iterate_trajectories(
    result: Mapping[str, Any],
) -> tuple[int, list[Any], list[Any], list[float], list[float]]:
    """Return the evaluated primal and averaged trajectories with norms."""
    iterations = int(result["iterations"])
    evaluated_x = list(result["x"][:iterations])
    evaluated_x_hat = list(result["x_hat"][1: iterations + 1])
    x_norm = [_value_norm(x_k) for x_k in evaluated_x]
    x_hat_norm = [_value_norm(x_hat_k) for x_hat_k in evaluated_x_hat]
    return iterations, evaluated_x, evaluated_x_hat, x_norm, x_hat_norm


def _objective_metadata(objective: _HasObjectiveMetadata) -> dict[str, Any]:
    """Return JSON-friendly metadata shared by all objective-backed summaries."""
    return {
        "objective_id": objective.id,
        "objective_family": objective.family,
        "objective_name": objective.name,
        "objective_params": _to_jsonable(dict(objective.params)),
        "objective_dimension": objective.dimension,
        "L": objective.lipschitz_constant,
    }


def _known_minimum_metadata(objective: _HasKnownMinimum) -> dict[str, Any]:
    """Return the saved optimum metadata for objectives with known minima."""
    return {
        "objective_minimum_value": float(objective.minimum_value),
        "objective_minimizer": _to_jsonable(objective.minimizer),
        "objective_minimizer_norm": _value_norm(objective.minimizer),
    }


def evaluate_trajectory_metric(
    metric_fn: Callable[[Any], Any],
    evaluated_x: Sequence[Any],
    evaluated_x_hat: Sequence[Any],
    *,
    cast: Callable[[Any], Any] = float,
) -> tuple[list[Any], list[Any]]:
    """Evaluate one metric on primal and averaged trajectories."""
    metric_x = [cast(metric_fn(x_k)) for x_k in evaluated_x]
    metric_x_hat = [cast(metric_fn(x_hat_k)) for x_hat_k in evaluated_x_hat]
    return metric_x, metric_x_hat


def build_sda_summary_base(
    *,
    result: Mapping[str, Any],
    objective: _HasObjectiveMetadata,
    D: float,
    gamma: float,
    gamma_multiplier: float,
    prox_center: float | np.ndarray,
    iterations: int,
    x_norm: Sequence[float],
    x_hat_norm: Sequence[float],
) -> dict[str, Any]:
    """Return fields shared by SDA summaries across scripts."""
    x = _to_jsonable(result["x"])
    x_hat = _to_jsonable(result["x_hat"])
    normalized_gap = [gap_k / (k + 1) for k, gap_k in enumerate(result["gap"])]

    return {
        "method": "sda",
        "solver_name": "sda",
        "solver_family": "simple_dual_averaging",
        "converged": result["converged"],
        "iterations": iterations,
        "total_runtime_seconds": float(result["total_runtime_seconds"]),
        "avg_iteration_time_seconds": float(result["avg_iteration_time_seconds"]),
        **_objective_metadata(objective),
        "restrict_to_fd": result["restrict_to_fd"],
        "prox_center": _to_jsonable(prox_center),
        "D": D,
        "sigma": 1.0,
        "gamma": gamma,
        "gamma_multiplier": gamma_multiplier,
        "gamma_star": gamma_star(D, L=objective.lipschitz_constant),
        "x": x,
        "x_hat": x_hat,
        "x_norm": list(x_norm),
        "x_hat_norm": list(x_hat_norm),
        "s": _to_jsonable(result["s"]),
        "g": _to_jsonable(result["g"]),
        "B_hat": _to_jsonable(result["B_hat"]),
        "B": _to_jsonable(result["B"]),
        "gap_accum": _to_jsonable(result["gap_accum"]),
        "gap": _to_jsonable(result["gap"]),
        "normalized_gap": normalized_gap,
        "final_x": x[-1],
        "final_x_hat": x_hat[-1],
        "final_x_norm": _value_norm(result["x"][-1]),
        "final_x_hat_norm": _value_norm(result["x_hat"][-1]),
        "final_normalized_gap": normalized_gap[-1],
    }


def build_subgradient_summary_base(
    *,
    result: Mapping[str, Any],
    objective: _HasObjectiveMetadata,
    D: float,
    alpha: float,
    prox_center: float | np.ndarray,
    iterations: int,
    x_norm: Sequence[float],
    x_hat_norm: Sequence[float],
) -> dict[str, Any]:
    """Return fields shared by projected subgradient summaries across scripts."""
    x = _to_jsonable(result["x"])
    x_hat = _to_jsonable(result["x_hat"])

    return {
        "method": "subgradient",
        "solver_name": "subgradient",
        "solver_family": "projected_subgradient",
        "converged": False,
        "iterations": iterations,
        "total_runtime_seconds": float(result["total_runtime_seconds"]),
        "avg_iteration_time_seconds": float(result["avg_iteration_time_seconds"]),
        **_objective_metadata(objective),
        "restrict_to_fd": result["restrict_to_fd"],
        "prox_center": _to_jsonable(prox_center),
        "D": D,
        "alpha": alpha,
        "x": x,
        "x_hat": x_hat,
        "x_norm": list(x_norm),
        "x_hat_norm": list(x_hat_norm),
        "g": _to_jsonable(result["g"]),
        "g_norm": [_value_norm(g_k) for g_k in result["g"]],
        "alpha_k": _to_jsonable(result["alpha"]),
        "final_x": x[-1],
        "final_x_hat": x_hat[-1],
        "final_x_norm": _value_norm(result["x"][-1]),
        "final_x_hat_norm": _value_norm(result["x_hat"][-1]),
    }


def run_single_sda_experiment(
    *,
    objective: _HasSDAInputs,
    D: float,
    eps: float,
    gamma_multiplier: float,
    max_iter: int,
    summarize_run: Callable[..., SummaryResult],
    restrict_to_fd: bool = False,
    prox_center: float | np.ndarray = 0.0,
) -> SummaryResult:
    """Run one SDA configuration and summarize it with the provided callback."""
    solver = SDA(prox_center=prox_center)
    base_gamma = gamma_star(D=D, L=objective.lipschitz_constant)
    gamma = gamma_multiplier * base_gamma

    result = solver.run(
        gamma=gamma,
        D=D,
        eps=eps,
        subgradient_oracle=objective.subgradient,
        max_iter=max_iter,
        restrict_to_fd=restrict_to_fd,
    )

    return summarize_run(
        result,
        objective=objective,
        D=D,
        gamma=gamma,
        gamma_multiplier=gamma_multiplier,
        prox_center=prox_center,
    )


def run_single_subgradient_experiment(
    *,
    objective: _HasSDAInputs,
    D: float,
    alpha: float,
    max_iter: int,
    summarize_run: Callable[..., SummaryResult],
    restrict_to_fd: bool = False,
    prox_center: float | np.ndarray = 0.0,
) -> SummaryResult:
    """Run one projected subgradient configuration and summarize it."""
    solver = SubgradientMethod(prox_center=prox_center)
    result = solver.run(
        gamma=alpha,
        D=D,
        subgradient_oracle=objective.subgradient,
        max_iter=max_iter,
        restrict_to_fd=restrict_to_fd,
    )

    return summarize_run(
        result,
        objective=objective,
        D=D,
        alpha=alpha,
        prox_center=prox_center,
    )


def save_results(
    runs: list[dict[str, Any]],
    output_dir: Path,
    run_id: str | None = None,
) -> Path:
    """Append run summaries to `results.json`, creating it if needed."""
    output_path = output_dir / "results.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError(f"Existing results file must contain a JSON array: {output_path}")
    else:
        existing = []

    if run_id is not None and any(
        isinstance(existing_run, dict) and existing_run.get("run_id") == run_id
        for existing_run in existing
    ):
        return output_path

    runs_to_append = []
    for run in runs:
        run_payload = dict(run)
        if run_id is not None:
            run_payload["run_id"] = run_id
        runs_to_append.append(run_payload)

    combined = [*existing, *runs_to_append]
    output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return output_path
