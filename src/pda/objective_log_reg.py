from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
ObjectiveValue = Union[float, FloatArray]
ObjectiveFn = Callable[[ObjectiveValue], float]
SubgradientFn = Callable[[ObjectiveValue], ObjectiveValue]
JsonValue = Union[None, bool, int, float, str, list[Any], dict[str, Any]]
AccuracyFn = Callable[[ObjectiveValue], float]
CountFn = Callable[[ObjectiveValue], int]


@dataclass(frozen=True)
class LogisticRegressionObjectiveDefinition:
    """Dataset-backed logistic-regression objective used by experiment scripts."""

    id: str
    family: str
    name: str
    params: Mapping[str, JsonValue]
    dimension: int
    lipschitz_constant: float
    X_train: FloatArray
    y_train: FloatArray
    X_test: FloatArray
    y_test: FloatArray
    objective: ObjectiveFn
    subgradient: SubgradientFn
    train_loss: ObjectiveFn
    test_loss: ObjectiveFn
    test_accuracy: AccuracyFn
    nonzero_count: CountFn


def _sign_with_zero(values: FloatArray) -> FloatArray:
    """Return a deterministic subgradient for the L1 norm."""
    signs = np.sign(values).astype(float)
    signs[np.isclose(values, 0.0)] = 0.0
    return signs


def _sigmoid(values: FloatArray) -> FloatArray:
    """Numerically stable sigmoid."""
    positive = values >= 0.0
    negative = ~positive
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    return result


def _as_vector(value: ObjectiveValue, *, dimension: int) -> FloatArray:
    """Convert an input to a fixed-dimensional vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,):
        raise ValueError(
            f"Expected a vector of shape {(dimension,)}, got {array.shape}."
        )
    return array


def _resolve_dataset_path(dataset_path: Path) -> Path:
    """Resolve a dataset path, falling back to the project's data directory."""
    if dataset_path.exists():
        return dataset_path.resolve()

    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "data" / dataset_path
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def _candidate_label_columns(headers: Sequence[str]) -> list[str]:
    normalized_to_original = {header.strip().lower(): header for header in headers}
    ordered_candidates = [
        "y",
        "target",
        "label",
        "labels",
        "class",
    ]
    return [
        normalized_to_original[name]
        for name in ordered_candidates
        if name in normalized_to_original
    ]


def _is_missing_csv_value(value: str | None) -> bool:
    return value is None or value == ""


def load_binary_classification_dataset(dataset_path: Path) -> tuple[FloatArray, FloatArray]:
    """Load a CSV dataset with numeric features and binary labels in {0, 1}."""
    resolved_path = _resolve_dataset_path(dataset_path)

    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset {resolved_path} is missing a header row.")
        if not reader.fieldnames:
            raise ValueError(f"Dataset {resolved_path} does not contain any columns.")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Dataset {resolved_path} does not contain any data rows.")

    label_candidates = _candidate_label_columns(reader.fieldnames)
    label_column = label_candidates[0] if label_candidates else reader.fieldnames[-1]
    feature_columns = [
        column_name
        for column_name in reader.fieldnames
        if column_name != label_column
    ]

    retained_feature_columns: list[str] = []
    for column_name in feature_columns:
        is_numeric_column = True
        for row in rows:
            raw_value = row[column_name]
            if _is_missing_csv_value(raw_value):
                continue
            try:
                float(raw_value)
            except ValueError:
                is_numeric_column = False
                break
        if is_numeric_column:
            retained_feature_columns.append(column_name)

    if not retained_feature_columns:
        raise ValueError(
            f"No numeric feature columns remain in dataset {resolved_path} "
            "after dropping non-numeric columns."
        )

    features: list[list[float]] = []
    labels: list[float] = []

    for row_index, row in enumerate(rows, start=2):
        raw_label = row[label_column]
        if _is_missing_csv_value(raw_label):
            continue

        try:
            label_value = float(raw_label)
        except ValueError as exc:
            raise ValueError(
                f"Label column '{label_column}' must be numeric; "
                f"got {raw_label!r} at line {row_index}."
            ) from exc

        feature_row: list[float] = []
        drop_row = False
        for column_name in retained_feature_columns:
            raw_value = row[column_name]
            if _is_missing_csv_value(raw_value):
                drop_row = True
                break
            try:
                feature_row.append(float(raw_value))
            except ValueError as exc:
                raise ValueError(
                    f"Feature column '{column_name}' was inferred as numeric but "
                    f"contains non-numeric value {raw_value!r} at line {row_index}."
                ) from exc

        if drop_row:
            continue

        features.append(feature_row)
        labels.append(label_value)

    if not features:
        raise ValueError(
            f"Dataset {resolved_path} does not contain any usable data rows "
            "after dropping missing values."
        )

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)

    unique_labels = set(np.unique(y).tolist())
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(
            f"Expected binary labels in {{0, 1}}, got {sorted(unique_labels)}."
        )

    if len(unique_labels) < 2:
        raise ValueError("Dataset must contain both label classes 0 and 1.")

    return X, y


def train_test_split(
    X: FloatArray,
    y: FloatArray,
    *,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Create a reproducible stratified train/test split."""
    if X.ndim != 2:
        raise ValueError(f"Expected X to be a matrix, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"Expected y to be a vector, got shape {y.shape}.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if len(X) < 2:
        raise ValueError("Need at least two samples to create train and test splits.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must satisfy 0 < test_size < 1.")

    test_count = max(1, int(round(len(X) * test_size)))
    if test_count >= len(X):
        test_count = len(X) - 1

    classes, class_counts = np.unique(y, return_counts=True)
    if np.any(class_counts < 2):
        raise ValueError(
            "Stratified split requires at least two samples in each class."
        )

    class_count = len(classes)
    if test_count < class_count or test_count > len(X) - class_count:
        raise ValueError(
            "Stratified split cannot preserve every class in both train and test sets "
            f"with test_size={test_size}."
        )

    raw_test_counts = class_counts.astype(float) * test_size
    min_test_counts = np.ones_like(class_counts, dtype=int)
    max_test_counts = class_counts - 1
    per_class_test_counts = np.floor(raw_test_counts).astype(int)
    per_class_test_counts = np.maximum(per_class_test_counts, min_test_counts)
    per_class_test_counts = np.minimum(per_class_test_counts, max_test_counts)

    while int(np.sum(per_class_test_counts)) < test_count:
        candidates = np.where(per_class_test_counts < max_test_counts)[0]
        if len(candidates) == 0:
            raise ValueError(
                "Unable to allocate a stratified test split that preserves every class."
            )
        best_index = int(
            candidates[np.argmax(raw_test_counts[candidates] - per_class_test_counts[candidates])]
        )
        per_class_test_counts[best_index] += 1

    while int(np.sum(per_class_test_counts)) > test_count:
        candidates = np.where(per_class_test_counts > min_test_counts)[0]
        if len(candidates) == 0:
            raise ValueError(
                "Unable to allocate a stratified train split that preserves every class."
            )
        best_index = int(
            candidates[np.argmax(per_class_test_counts[candidates] - raw_test_counts[candidates])]
        )
        per_class_test_counts[best_index] -= 1

    rng = np.random.default_rng(seed)
    test_index_groups: list[FloatArray] = []
    train_index_groups: list[FloatArray] = []
    all_indices = np.arange(len(X), dtype=int)

    for class_value, class_test_count in zip(classes, per_class_test_counts):
        class_indices = all_indices[y == class_value]
        shuffled_indices = rng.permutation(class_indices)
        test_index_groups.append(shuffled_indices[:class_test_count])
        train_index_groups.append(shuffled_indices[class_test_count:])

    test_indices = np.concatenate(test_index_groups)
    train_indices = np.concatenate(train_index_groups)
    test_indices = rng.permutation(test_indices)
    train_indices = rng.permutation(train_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_features(
    X_train: FloatArray,
    X_test: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Standardize features with train-set statistics only."""
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Expected train and test features to be matrices.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test features must have the same number of columns.")

    means = np.mean(X_train, axis=0)
    scales = np.std(X_train, axis=0)
    scales[np.isclose(scales, 0.0)] = 1.0

    return (X_train - means) / scales, (X_test - means) / scales


def add_bias_column(X: FloatArray) -> FloatArray:
    """Append a column of ones to the feature matrix."""
    return np.column_stack([X, np.ones(len(X), dtype=float)])


def _weights_without_bias(weights: FloatArray) -> FloatArray:
    """Return only feature weights, excluding the final bias coordinate."""
    return weights[:-1]


def build_logistic_regression_objective(
    dataset_path: Path,
    *,
    lasso: bool = False,
    lasso_lambda: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
) -> LogisticRegressionObjectiveDefinition:
    """Create a train/test logistic-regression objective from a CSV dataset."""
    if lasso_lambda < 0.0:
        raise ValueError("lasso_lambda must be nonnegative.")

    resolved_path = _resolve_dataset_path(dataset_path)
    X_raw, y = load_binary_classification_dataset(resolved_path)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        seed=seed,
    )
    X_train_scaled, X_test_scaled = standardize_features(X_train_raw, X_test_raw)
    X_train = add_bias_column(X_train_scaled)
    X_test = add_bias_column(X_test_scaled)
    dimension = X_train.shape[1]
    train_sample_count = X_train.shape[0]

    def empirical_loss(weights: ObjectiveValue, *, X_data: FloatArray, y_data: FloatArray) -> float:
        weights_array = _as_vector(weights, dimension=dimension)
        logits = X_data @ weights_array
        loss = np.mean(np.logaddexp(0.0, logits) - y_data * logits)
        if not lasso:
            return float(loss)
        return float(
            loss
            + (lasso_lambda / train_sample_count)
            * np.linalg.norm(_weights_without_bias(weights_array), ord=1)
        )

    def subgradient(weights: ObjectiveValue) -> FloatArray:
        weights_array = _as_vector(weights, dimension=dimension)
        logits = X_train @ weights_array
        probabilities = _sigmoid(logits)
        gradient = (X_train.T @ (probabilities - y_train)) / train_sample_count
        if lasso:
            penalty_gradient = np.zeros_like(weights_array)
            penalty_gradient[:-1] = _sign_with_zero(_weights_without_bias(weights_array))
            gradient = gradient + (lasso_lambda / train_sample_count) * penalty_gradient
        return gradient

    def test_accuracy(weights: ObjectiveValue) -> float:
        weights_array = _as_vector(weights, dimension=dimension)
        probabilities = _sigmoid(X_test @ weights_array)
        predictions = (probabilities >= 0.5).astype(float)
        return float(np.mean(predictions == y_test))

    def nonzero_count(weights: ObjectiveValue) -> int:
        weights_array = _as_vector(weights, dimension=dimension)
        return int(np.count_nonzero(~np.isclose(weights_array, 0.0)))

    lipschitz_constant = float(np.mean(np.linalg.norm(X_train, axis=1)))
    if lasso:
        lipschitz_constant += float((lasso_lambda / train_sample_count) * np.sqrt(dimension))

    if lipschitz_constant <= 0.0:
        raise ValueError("Estimated Lipschitz constant must be positive.")

    objective_id = f"{resolved_path.stem}_{'lasso' if lasso else 'logreg'}"
    objective_name = "Lasso logistic regression" if lasso else "Logistic regression"

    return LogisticRegressionObjectiveDefinition(
        id=objective_id,
        family="lasso_logistic_regression" if lasso else "logistic_regression",
        name=f"{objective_name} on {resolved_path.name}",
        params={
            "dataset_path": str(resolved_path),
            "lasso": lasso,
            "lasso_lambda": float(lasso_lambda),
            "test_size": float(test_size),
            "seed": int(seed),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        dimension=dimension,
        lipschitz_constant=lipschitz_constant,
        X_train=X_train,
        y_train=y_train.copy(),
        X_test=X_test,
        y_test=y_test.copy(),
        objective=lambda weights: empirical_loss(weights, X_data=X_train, y_data=y_train),
        subgradient=subgradient,
        train_loss=lambda weights: empirical_loss(weights, X_data=X_train, y_data=y_train),
        test_loss=lambda weights: empirical_loss(weights, X_data=X_test, y_data=y_test),
        test_accuracy=test_accuracy,
        nonzero_count=nonzero_count,
    )
