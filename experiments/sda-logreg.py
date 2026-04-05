from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys
from typing import Dict
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import (  # noqa
    _prepare_iterate_trajectories,
    _to_jsonable,
    _zero_prox_center,
    build_run_id,
    build_sda_summary_base,
    build_subgradient_summary_base,
    evaluate_trajectory_metric,
    run_single_sda_experiment,
    run_single_subgradient_experiment,
    save_results,
)
from pda import (  # noqa
    LogisticRegressionObjectiveDefinition,
    SDAResult,
    SubgradientResult,
    build_logistic_regression_objective,
)


def summarize_sda_run(
    result: SDAResult,
    *,
    objective: LogisticRegressionObjectiveDefinition,
    D: float,
    gamma: float,
    gamma_multiplier: float,
    prox_center: np.ndarray,
) -> Dict[str, object]:
    """Build a JSON-friendly summary for a single SDA run."""
    iterations, evaluated_x, evaluated_x_hat, x_norm, x_hat_norm = _prepare_iterate_trajectories(
        result
    )
    train_loss_x, train_loss_x_hat = evaluate_trajectory_metric(
        objective.train_loss, evaluated_x, evaluated_x_hat
    )
    test_loss_x, test_loss_x_hat = evaluate_trajectory_metric(
        objective.test_loss, evaluated_x, evaluated_x_hat
    )
    test_accuracy_x, test_accuracy_x_hat = evaluate_trajectory_metric(
        objective.test_accuracy, evaluated_x, evaluated_x_hat
    )
    nonzero_count_x, nonzero_count_x_hat = evaluate_trajectory_metric(
        objective.nonzero_count, evaluated_x, evaluated_x_hat, cast=int
    )

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
        "train_loss_x": train_loss_x,
        "train_loss_x_hat": train_loss_x_hat,
        "test_loss_x": test_loss_x,
        "test_loss_x_hat": test_loss_x_hat,
        "test_accuracy_x": test_accuracy_x,
        "test_accuracy_x_hat": test_accuracy_x_hat,
        "nonzero_count_x": nonzero_count_x,
        "nonzero_count_x_hat": nonzero_count_x_hat,
        "final_train_loss_x": train_loss_x[-1],
        "final_train_loss_x_hat": train_loss_x_hat[-1],
        "final_test_loss_x": test_loss_x[-1],
        "final_test_loss_x_hat": test_loss_x_hat[-1],
        "final_test_accuracy_x": test_accuracy_x[-1],
        "final_test_accuracy_x_hat": test_accuracy_x_hat[-1],
        "final_nonzero_count_x": nonzero_count_x[-1],
        "final_nonzero_count_x_hat": nonzero_count_x_hat[-1],
    }


def summarize_subgradient_run(
    result: SubgradientResult,
    *,
    objective: LogisticRegressionObjectiveDefinition,
    D: float,
    alpha: float,
    prox_center: np.ndarray,
) -> Dict[str, object]:
    """Build a JSON-friendly summary for a single subgradient run."""
    iterations, evaluated_x, evaluated_x_hat, x_norm, x_hat_norm = _prepare_iterate_trajectories(
        result
    )
    train_loss_x, train_loss_x_hat = evaluate_trajectory_metric(
        objective.train_loss, evaluated_x, evaluated_x_hat
    )
    test_loss_x, test_loss_x_hat = evaluate_trajectory_metric(
        objective.test_loss, evaluated_x, evaluated_x_hat
    )
    test_accuracy_x, test_accuracy_x_hat = evaluate_trajectory_metric(
        objective.test_accuracy, evaluated_x, evaluated_x_hat
    )
    nonzero_count_x, nonzero_count_x_hat = evaluate_trajectory_metric(
        objective.nonzero_count, evaluated_x, evaluated_x_hat, cast=int
    )

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
        "train_loss_x": train_loss_x,
        "train_loss_x_hat": train_loss_x_hat,
        "test_loss_x": test_loss_x,
        "test_loss_x_hat": test_loss_x_hat,
        "test_accuracy_x": test_accuracy_x,
        "test_accuracy_x_hat": test_accuracy_x_hat,
        "nonzero_count_x": nonzero_count_x,
        "nonzero_count_x_hat": nonzero_count_x_hat,
        "final_train_loss_x": train_loss_x[-1],
        "final_train_loss_x_hat": train_loss_x_hat[-1],
        "final_test_loss_x": test_loss_x[-1],
        "final_test_loss_x_hat": test_loss_x_hat[-1],
        "final_test_accuracy_x": test_accuracy_x[-1],
        "final_test_accuracy_x_hat": test_accuracy_x_hat[-1],
        "final_nonzero_count_x": nonzero_count_x[-1],
        "final_nonzero_count_x_hat": nonzero_count_x_hat[-1],
    }


def summarize_sklearn_run(
    *,
    objective: LogisticRegressionObjectiveDefinition,
    D: float,
    max_iter: int,
) -> Dict[str, object]:
    """Fit a sklearn baseline and return a JSON-friendly summary."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    X_train_raw = objective.X_train[:, :-1]
    penalty = None
    solver_name = "saga"
    model_kwargs: Dict[str, object] = {
        "solver": solver_name,
        "penalty": penalty,
        "fit_intercept": True,
        "max_iter": max_iter,
        "random_state": int(objective.params["seed"]),
    }

    if bool(objective.params["lasso"]):
        lasso_lambda = float(objective.params["lasso_lambda"])
        penalty = "l1"
        if lasso_lambda <= 0.0:
            lasso_lambda = 1e-12
        model_kwargs = {
            "solver": solver_name,
            "penalty": penalty,
            "C": 1.0 / lasso_lambda,
            "fit_intercept": True,
            "max_iter": max_iter,
            "random_state": int(objective.params["seed"]),
        }

    model = LogisticRegression(**model_kwargs)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        start_time = perf_counter()
        model.fit(X_train_raw, objective.y_train.astype(int))
        total_runtime_seconds = perf_counter() - start_time

    iterations = int(np.max(np.asarray(model.n_iter_, dtype=int)))
    avg_iteration_time_seconds = (
        float(total_runtime_seconds / iterations) if iterations > 0 else 0.0
    )
    warning_messages = [
        str(warning.message)
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]
    converged = not warning_messages

    coefficient_vector = np.asarray(model.coef_[0], dtype=float)
    intercept = float(model.intercept_[0])
    parameter_vector = np.concatenate([coefficient_vector, np.asarray([intercept])])
    train_loss = float(objective.train_loss(parameter_vector))
    test_loss = float(objective.test_loss(parameter_vector))
    test_accuracy = float(objective.test_accuracy(parameter_vector))
    nonzero_count = int(objective.nonzero_count(parameter_vector))

    return {
        "method": "sklearn",
        "solver_name": solver_name,
        "solver_family": "sklearn_logistic_regression",
        "converged": converged,
        "iterations": iterations,
        "total_runtime_seconds": float(total_runtime_seconds),
        "avg_iteration_time_seconds": avg_iteration_time_seconds,
        "objective_id": objective.id,
        "objective_family": objective.family,
        "objective_name": objective.name,
        "objective_params": _to_jsonable(dict(objective.params)),
        "objective_dimension": objective.dimension,
        "restrict_to_fd": False,
        "D": D,
        "L": objective.lipschitz_constant,
        "sklearn_penalty": penalty,
        "sklearn_fit_intercept": True,
        "sklearn_max_iter": int(max_iter),
        "sklearn_warning_messages": warning_messages,
        "sklearn_coef": _to_jsonable(coefficient_vector),
        "sklearn_intercept": intercept,
        "sklearn_parameter_vector": _to_jsonable(parameter_vector),
        "final_parameter_vector": _to_jsonable(parameter_vector),
        "final_parameter_norm": float(np.linalg.norm(parameter_vector)),
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "final_nonzero_count": nonzero_count,
    }


def print_summary(run: Dict[str, object]) -> None:
    """Print a compact textual summary of one experiment run."""
    params = run["objective_params"]
    method = str(run["method"])
    if method == "sda":
        step_display = f"{run['gamma_multiplier']:g} gamma*"
        title = "SDA"
    elif method == "subgradient":
        step_display = f"alpha={run['alpha']:g}"
        title = "Subgradient"
    else:
        step_display = "-"
        title = "sklearn"

    train_loss = run.get("final_train_loss_x_hat", run.get("final_train_loss"))
    test_accuracy = run.get("final_test_accuracy_x_hat", run.get("final_test_accuracy"))
    nonzero_count = run.get("final_nonzero_count_x_hat", run.get("final_nonzero_count"))

    print(title)
    print(f"objective_id: {run['objective_id']}")
    print(f"dataset: {params['dataset_path']}")
    print(f"lasso: {params['lasso']}")
    print(f"train_samples: {params['train_samples']}")
    print(f"test_samples: {params['test_samples']}")
    print(
        "method       solver       D      step_param    converged  iterations  "
        "runtime(s)  avg_iter(s)  train_loss  test_acc  nnz"
    )
    print(
        f"{run['method']:<12} "
        f"{run['solver_name']:<12} "
        f"{run['D']:<6g} "
        f"{step_display:<13} "
        f"{str(run['converged']):<10} "
        f"{run['iterations']:<11d} "
        f"{run['total_runtime_seconds']:<11.8f} "
        f"{run['avg_iteration_time_seconds']:<12.8f} "
        f"{train_loss:<11.8f} "
        f"{test_accuracy:<10.8f} "
        f"{nonzero_count}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the logistic-regression experiment."""
    parser = argparse.ArgumentParser(
        description="Run SDA, projected subgradient, and sklearn logistic regression from a CSV dataset."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset CSV path. If a relative path does not exist, it is resolved under the data/ directory.",
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
        default=1.0,
        help="Gamma multiplier of the theoretical gamma*.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Subgradient alpha numerator used in alpha_k = alpha / sqrt(k + 1).",
    )
    parser.add_argument("--eps", type=float, default=1e-3, help="Stopping tolerance.")
    parser.add_argument(
        "-i",
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of iterations per run.",
    )
    parser.add_argument(
        "--lasso",
        action="store_true",
        help="Include an L1 penalty, solving the lasso logistic objective.",
    )
    parser.add_argument(
        "--lambda",
        dest="lasso_lambda",
        type=float,
        default=1.0,
        help="L1 penalty coefficient used when --lasso is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the 80/20 train/test split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used for JSON summaries. Default: outputs/<script-stem>/<dataset-stem>/<mode>.",
    )
    parser.add_argument(
        "-r",
        "--restrict-to-fd",
        action="store_true",
        help="Project each primal iterate onto F_D, solving the problem restricted to the Euclidean D-ball.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the logistic-regression experiment workflow."""
    args = parse_args()
    run_id = build_run_id(args)
    objective = build_logistic_regression_objective(
        args.dataset,
        lasso=args.lasso,
        lasso_lambda=args.lasso_lambda,
        test_size=args.test_size,
        seed=args.seed,
    )
    prox_center = _zero_prox_center(objective)
    mode_name = "lasso" if args.lasso else "logreg"
    dataset_stem = Path(str(objective.params["dataset_path"])).stem
    output_dir = args.output_dir or (
        Path("outputs") / Path(__file__).stem / dataset_stem / mode_name
    )
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
    sklearn_run = summarize_sklearn_run(
        objective=objective,
        D=args.D,
        max_iter=args.max_iter,
    )
    runs = [sda_run, subgradient_run, sklearn_run]

    print_summary(sda_run)
    print()
    print_summary(subgradient_run)
    print()
    print_summary(sklearn_run)

    results_path = save_results(runs, output_dir, run_id=run_id)
    print(f"\nResults JSON path: {results_path}")


if __name__ == "__main__":
    main()
