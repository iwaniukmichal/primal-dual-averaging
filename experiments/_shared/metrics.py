from __future__ import annotations

from typing import Any, Callable

import numpy as np


def value_norm(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(abs(array))
    return float(np.linalg.norm(array))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(inner_value) for inner_value in value]
    return value


def trajectory_metric(
    metric_fn: Callable[[Any], Any],
    values: list[Any],
    *,
    cast: Callable[[Any], Any] = float,
) -> list[Any]:
    return [cast(metric_fn(value)) for value in values]


def gamma_star(D: float, L: float, sigma: float = 1.0) -> float:
    return float(L / np.sqrt(2.0 * sigma * D))


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float

