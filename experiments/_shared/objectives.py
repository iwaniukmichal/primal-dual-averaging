from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import PROJECT_ROOT
from pda import build_logistic_regression_objective, get_objective, list_objective_ids
from pda.objective_log_reg import add_bias_column, standardize_features, train_test_split
from data.generate_logistic_data import generate_dataset, write_csv


def registry_objective_ids(*, include_simplex: bool = True) -> list[str]:
    ids = list_objective_ids()
    if include_simplex:
        return ids
    return [objective_id for objective_id in ids if not objective_id.startswith("simplex_")]


def prox_center_for_objective(dimension: int, prox_fun: str) -> np.ndarray | float:
    if prox_fun == "entropy":
        return np.full(dimension, 1.0 / dimension, dtype=float)
    if dimension == 1:
        return 0.0
    return np.zeros(dimension, dtype=float)


def prox_weights_for_profile(dimension: int, profile: str) -> np.ndarray | None:
    if profile in {"", "none"}:
        return None
    if profile == "uniform":
        return np.ones(dimension, dtype=float)
    coordinate_scale = np.geomspace(0.1, 10.0, num=dimension)
    if profile == "coordinate_scale":
        return coordinate_scale
    if profile == "inverse_coordinate_scale":
        return 1.0 / coordinate_scale
    raise ValueError(f"Unknown prox weight profile: {profile}")


def build_registry_objective(objective_id: str) -> Any:
    return get_objective(objective_id)


def build_logreg_objective(
    dataset: str | Path,
    *,
    lasso: bool = False,
    lasso_lambda: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
) -> Any:
    return build_logistic_regression_objective(
        Path(dataset),
        lasso=lasso,
        lasso_lambda=lasso_lambda,
        test_size=test_size,
        seed=seed,
    )


def generate_temporary_logistic_dataset(
    output_dir: Path,
    *,
    n_samples: int,
    dimension: int,
    seed: int,
    flip_prob: float,
) -> Path:
    beta = np.linspace(-1.0, 1.0, dimension, dtype=float)
    if np.allclose(beta, 0.0):
        beta[0] = 1.0
    features, labels = generate_dataset(
        n_samples=n_samples,
        dimension=dimension,
        beta=beta,
        intercept=0.1,
        flip_prob=flip_prob,
        seed=seed,
    )
    output_path = output_dir / "generated_data" / (
        f"logistic_n{n_samples}_d{dimension}_seed{seed}_flip{flip_prob:g}.csv"
    )
    write_csv(output_path, features, labels)
    return output_path


def prepared_logistic_arrays(dataset: str | Path, *, test_size: float, seed: int) -> dict[str, Any]:
    """Return standardized train/test arrays for optional diagnostics."""
    from pda.objective_log_reg import load_binary_classification_dataset

    X_raw, y = load_binary_classification_dataset(Path(dataset))
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        seed=seed,
    )
    X_train_scaled, X_test_scaled = standardize_features(X_train_raw, X_test_raw)
    return {
        "X_train": add_bias_column(X_train_scaled),
        "X_test": add_bias_column(X_test_scaled),
        "y_train": y_train,
        "y_test": y_test,
    }


def data_path(name: str) -> Path:
    return PROJECT_ROOT / "data" / name

