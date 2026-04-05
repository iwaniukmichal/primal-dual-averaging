from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import (  # noqa
    _prepare_iterate_trajectories,
    _zero_prox_center,
    build_run_id,
    build_sda_summary_base,
    build_subgradient_summary_base,
    evaluate_trajectory_metric,
    gamma_star,
    run_single_sda_experiment,
    run_single_subgradient_experiment,
    save_results,
    _known_minimum_metadata,
)
from pda import (  # noqa
    ObjectiveDefinition,
    SDAResult,
    SubgradientResult,
    get_objective,
    list_objective_ids,
)


def _format_value(value: Any) -> str:
    """Format a scalar or vector value compactly for terminal output."""
    if isinstance(value, list):
        return np.array2string(np.asarray(value, dtype=float), precision=4, separator=", ")
    if isinstance(value, np.ndarray):
        return np.array2string(value.astype(float), precision=4, separator=", ")
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def summarize_sda_run(
    result: SDAResult,
    *,
    objective: ObjectiveDefinition,
    D: float,
    gamma: float,
    gamma_multiplier: float,
    prox_center: float | np.ndarray,
) -> Dict[str, Any]:
    """Build a JSON-friendly summary for a single SDA run."""
    iterations, evaluated_x, evaluated_x_hat, x_norm, x_hat_norm = _prepare_iterate_trajectories(
        result
    )
    f_x, f_x_hat = evaluate_trajectory_metric(objective.objective, evaluated_x, evaluated_x_hat)
    objective_minimum_value = float(objective.minimum_value)
    final_objective_x = float(objective.objective(result["x"][-1]))
    final_objective_x_hat = float(objective.objective(result["x_hat"][-1]))

    return {
        **build_sda_summary_base(
            result=result,
            objective=objective,
            D=D,
            gamma=gamma,
            gamma_multiplier=gamma_multiplier,
            prox_center=prox_center,
            iterations=iterations,
            x_norm=x_norm,
            x_hat_norm=x_hat_norm,
        ),
        **_known_minimum_metadata(objective),
        "f_x": f_x,
        "f_x_hat": f_x_hat,
        "final_objective_x": final_objective_x,
        "final_objective_x_hat": final_objective_x_hat,
        "final_objective_gap_x": final_objective_x - objective_minimum_value,
        "final_objective_gap_x_hat": final_objective_x_hat - objective_minimum_value,
    }


def summarize_subgradient_run(
    result: SubgradientResult,
    *,
    objective: ObjectiveDefinition,
    D: float,
    alpha: float,
    prox_center: float | np.ndarray,
) -> Dict[str, Any]:
    """Build a JSON-friendly summary for a single subgradient run."""
    iterations, evaluated_x, evaluated_x_hat, x_norm, x_hat_norm = _prepare_iterate_trajectories(
        result
    )
    f_x, f_x_hat = evaluate_trajectory_metric(objective.objective, evaluated_x, evaluated_x_hat)
    objective_gap_x = [value - float(objective.minimum_value) for value in f_x]
    objective_gap_x_hat = [value - float(objective.minimum_value) for value in f_x_hat]
    objective_minimum_value = float(objective.minimum_value)
    final_objective_x = float(objective.objective(result["x"][-1]))
    final_objective_x_hat = float(objective.objective(result["x_hat"][-1]))

    return {
        **build_subgradient_summary_base(
            result=result,
            objective=objective,
            D=D,
            alpha=alpha,
            prox_center=prox_center,
            iterations=iterations,
            x_norm=x_norm,
            x_hat_norm=x_hat_norm,
        ),
        **_known_minimum_metadata(objective),
        "f_x": f_x,
        "f_x_hat": f_x_hat,
        "objective_gap_x": objective_gap_x,
        "objective_gap_x_hat": objective_gap_x_hat,
        "final_objective_x": final_objective_x,
        "final_objective_x_hat": final_objective_x_hat,
        "final_objective_gap_x": final_objective_x - objective_minimum_value,
        "final_objective_gap_x_hat": final_objective_x_hat - objective_minimum_value,
    }


def print_sda_summary(run: Dict[str, Any]) -> None:
    """Print a compact textual summary of one SDA run."""
    print("SDA")
    print(f"objective_id: {run['objective_id']}")
    print(f"x*: {_format_value(run['objective_minimizer'])}")
    print(f"f*: {run['objective_minimum_value']:.8f}")
    print(
        "D      gamma/gamma*  converged  iterations  final_norm_gap  final_f(x_hat)  final_f(x)"
    )
    print(
        f"{run['D']:<6g} "
        f"{run['gamma_multiplier']:<13g} "
        f"{str(run['converged']):<10} "
        f"{run['iterations']:<11d} "
        f"{run['final_normalized_gap']:<15.8f} "
        f"{run['final_objective_x_hat']:<16.8f} "
        f"{run['final_objective_x']:.8f}"
    )


def print_subgradient_summary(run: Dict[str, Any]) -> None:
    """Print a compact textual summary of one subgradient run."""
    print("\nSubgradient")
    print(f"objective_id: {run['objective_id']}")
    print(f"x*: {_format_value(run['objective_minimizer'])}")
    print(f"f*: {run['objective_minimum_value']:.8f}")
    print("D      alpha      iterations  final_f(x_hat)  final_f(x)")
    print(
        f"{run['D']:<6g} "
        f"{run['alpha']:<10g} "
        f"{run['iterations']:<11d} "
        f"{run['final_objective_x_hat']:<16.8f} "
        f"{run['final_objective_x']:.8f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the combined Experiment 1 benchmark."""
    available_objectives = ", ".join(list_objective_ids())
    parser = argparse.ArgumentParser(
        description="Run Experiment 1 for Simple Dual Averaging and the projected subgradient method on registry objectives."
    )
    parser.add_argument(
        "-o",
        "--objective",
        required=True,
        choices=list_objective_ids(),
        help=f"Objective registry ID. Available values: {available_objectives}.",
    )
    parser.add_argument(
        "-d",
        "--D",
        type=float,
        required=True,
        help="Restriction parameter value to test.",
    )
    parser.add_argument(
        "-g",
        "--gamma-mult",
        type=float,
        required=True,
        help="Gamma multiplier of the theoretical gamma*.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Subgradient alpha numerator used in alpha_k = alpha / sqrt(k + 1).",
    )
    parser.add_argument("--eps", type=float, default=1e-3, help="Stopping tolerance for SDA.")
    parser.add_argument(
        "-i",
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of iterations per run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used for JSON summaries. Default: outputs/sda-bench/<objective-id>.",
    )
    parser.add_argument(
        "-r",
        "--restrict-to-fd",
        action="store_true",
        help="Project each primal iterate onto F_D, solving the problem restricted to the Euclidean D-ball.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the combined experiment workflow."""
    args = parse_args()
    run_id = build_run_id(args)
    objective = get_objective(args.objective)
    prox_center = _zero_prox_center(objective)
    output_dir = args.output_dir or (Path("outputs") / Path(__file__).stem / objective.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    sda_run = run_single_sda_experiment(
        objective=objective,
        D=args.D,
        eps=args.eps,
        gamma_multiplier=args.gamma_mult,
        max_iter=args.max_iter,
        summarize_run=summarize_sda_run,
        restrict_to_fd=args.restrict_to_fd,
        prox_center=prox_center,
    )
    subgradient_run = run_single_subgradient_experiment(
        objective=objective,
        D=args.D,
        alpha=args.alpha,
        max_iter=args.max_iter,
        summarize_run=summarize_subgradient_run,
        restrict_to_fd=args.restrict_to_fd,
        prox_center=prox_center,
    )
    runs = [sda_run, subgradient_run]

    print_sda_summary(sda_run)
    print_subgradient_summary(subgradient_run)

    results_path = save_results(runs, output_dir, run_id=run_id)
    print(f"\nResults JSON path: {results_path}")


if __name__ == "__main__":
    main()
