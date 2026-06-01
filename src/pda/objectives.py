from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Union

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
ObjectiveValue = Union[float, FloatArray]
ObjectiveFn = Callable[[ObjectiveValue], float]
SubgradientFn = Callable[[ObjectiveValue], ObjectiveValue]
JsonValue = Union[None, bool, int, float, str, list[Any], dict[str, Any]]


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Named objective preset used by experiment scripts."""

    id: str
    family: str
    name: str
    params: Mapping[str, JsonValue]
    dimension: int
    lipschitz_constant: float
    minimum_value: float
    minimizer: ObjectiveValue
    objective: ObjectiveFn
    subgradient: SubgradientFn


def _sign_with_zero(value: float) -> float:
    """Return a deterministic subgradient for |x|."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _to_float_array(values: Sequence[float]) -> FloatArray:
    """Convert a finite sequence to a float NumPy array."""
    return np.asarray(values, dtype=float)


def _as_scalar(value: ObjectiveValue) -> float:
    """Convert a scalar-like input to a Python float."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.shape == (1,):
        return float(array[0])
    raise ValueError(f"Expected a scalar input, got shape {array.shape}.")


def _as_vector(value: ObjectiveValue, *, dimension: int) -> FloatArray:
    """Convert an input to a fixed-dimensional vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,):
        raise ValueError(
            f"Expected a vector of shape {(dimension,)}, got {array.shape}."
        )
    return array


def _json_list(values: Sequence[float]) -> list[float]:
    """Convert a numeric sequence to a JSON-friendly float list."""
    return [float(value) for value in values]


def _make_abs_shift_objective(objective_id: str, *, a: float) -> ObjectiveDefinition:
    def objective(x: ObjectiveValue) -> float:
        return float(abs(_as_scalar(x) - a))

    def subgradient(x: ObjectiveValue) -> float:
        return _sign_with_zero(_as_scalar(x) - a)

    return ObjectiveDefinition(
        id=objective_id,
        family="absolute_shift",
        name=f"Absolute shift objective with a={a:g}",
        params={"a": float(a)},
        dimension=1,
        lipschitz_constant=1.0,
        minimum_value=0.0,
        minimizer=float(a),
        objective=objective,
        subgradient=subgradient,
    )


def _make_weighted_l1_shift_1d(
    objective_id: str,
    *,
    weights: Sequence[float],
    shifts: Sequence[float],
) -> ObjectiveDefinition:
    if len(weights) != len(shifts):
        raise ValueError("weights and shifts must have the same length.")

    weights_array = _to_float_array(weights)
    shifts_array = _to_float_array(shifts)

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_scalar(x)
        return float(np.sum(weights_array * np.abs(x_value - shifts_array)))

    def subgradient(x: ObjectiveValue) -> float:
        x_value = _as_scalar(x)
        return float(
            np.sum(
                weights_array
                * np.vectorize(_sign_with_zero, otypes=[float])(x_value - shifts_array)
            )
        )

    minimizer = float(np.median(shifts_array))

    return ObjectiveDefinition(
        id=objective_id,
        family="weighted_l1_shift",
        name="Weighted 1D L1 shift objective",
        params={
            "weights": _json_list(weights_array),
            "shifts": _json_list(shifts_array),
        },
        dimension=1,
        lipschitz_constant=float(np.sum(np.abs(weights_array))),
        minimum_value=float(objective(minimizer)),
        minimizer=minimizer,
        objective=objective,
        subgradient=subgradient,
    )


def _make_max_affine_1d(
    objective_id: str,
    *,
    slopes: Sequence[float],
    intercepts: Sequence[float],
) -> ObjectiveDefinition:
    if len(slopes) != len(intercepts):
        raise ValueError("slopes and intercepts must have the same length.")

    slopes_array = _to_float_array(slopes)
    intercepts_array = _to_float_array(intercepts)

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_scalar(x)
        return float(np.max(slopes_array * x_value + intercepts_array))

    def subgradient(x: ObjectiveValue) -> float:
        x_value = _as_scalar(x)
        values = slopes_array * x_value + intercepts_array
        active_index = int(np.argmax(values))
        return float(slopes_array[active_index])

    candidate_points = []
    for left_index in range(len(slopes_array)):
        for right_index in range(left_index + 1, len(slopes_array)):
            slope_diff = slopes_array[left_index] - slopes_array[right_index]
            if slope_diff == 0.0:
                continue
            candidate_points.append(
                float(
                    (intercepts_array[right_index] - intercepts_array[left_index])
                    / slope_diff
                )
            )
    candidate_points.extend([-1e6, 1e6])
    minimizer = min(candidate_points, key=objective)

    return ObjectiveDefinition(
        id=objective_id,
        family="max_affine",
        name="1D max-affine objective",
        params={
            "slopes": _json_list(slopes_array),
            "intercepts": _json_list(intercepts_array),
        },
        dimension=1,
        lipschitz_constant=float(np.max(np.abs(slopes_array))),
        minimum_value=float(objective(minimizer)),
        minimizer=minimizer,
        objective=objective,
        subgradient=subgradient,
    )


def _make_weighted_l1_shift_nd(
    objective_id: str,
    *,
    shifts: Sequence[float],
    weights: Sequence[float],
) -> ObjectiveDefinition:
    if len(shifts) != len(weights):
        raise ValueError("shifts and weights must have the same length.")

    shifts_array = _to_float_array(shifts)
    weights_array = _to_float_array(weights)
    dimension = len(shifts_array)

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_vector(x, dimension=dimension)
        return float(np.sum(weights_array * np.abs(x_value - shifts_array)))

    def subgradient(x: ObjectiveValue) -> FloatArray:
        x_value = _as_vector(x, dimension=dimension)
        diffs = x_value - shifts_array
        signs = np.array([_sign_with_zero(float(diff)) for diff in diffs], dtype=float)
        return weights_array * signs

    minimizer = shifts_array.copy()

    return ObjectiveDefinition(
        id=objective_id,
        family="weighted_l1_shift",
        name=f"{dimension}D weighted L1 shift objective",
        params={
            "weights": _json_list(weights_array),
            "shifts": _json_list(shifts_array),
        },
        dimension=dimension,
        lipschitz_constant=float(np.linalg.norm(weights_array)),
        minimum_value=float(objective(minimizer)),
        minimizer=minimizer.copy(),
        objective=objective,
        subgradient=subgradient,
    )


def _make_linf_shift_nd(
    objective_id: str,
    *,
    shifts: Sequence[float],
) -> ObjectiveDefinition:
    shifts_array = _to_float_array(shifts)
    dimension = len(shifts_array)

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_vector(x, dimension=dimension)
        return float(np.max(np.abs(x_value - shifts_array)))

    def subgradient(x: ObjectiveValue) -> FloatArray:
        x_value = _as_vector(x, dimension=dimension)
        diffs = x_value - shifts_array
        active_index = int(np.argmax(np.abs(diffs)))
        gradient = np.zeros(dimension, dtype=float)
        gradient[active_index] = _sign_with_zero(float(diffs[active_index]))
        return gradient

    minimizer = shifts_array.copy()

    return ObjectiveDefinition(
        id=objective_id,
        family="linf_shift",
        name=f"{dimension}D L-infinity shift objective",
        params={"shifts": _json_list(shifts_array)},
        dimension=dimension,
        lipschitz_constant=1.0,
        minimum_value=float(objective(minimizer)),
        minimizer=minimizer.copy(),
        objective=objective,
        subgradient=subgradient,
    )


def _make_max_affine_nd(
    objective_id: str,
    *,
    gradients: Sequence[Sequence[float]],
    intercepts: Sequence[float],
) -> ObjectiveDefinition:
    gradients_array = np.asarray(gradients, dtype=float)
    intercepts_array = _to_float_array(intercepts)

    if gradients_array.ndim != 2:
        raise ValueError("gradients must define a matrix.")
    if gradients_array.shape[0] != len(intercepts_array):
        raise ValueError("Need one intercept per affine piece.")

    dimension = gradients_array.shape[1]

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_vector(x, dimension=dimension)
        return float(np.max(gradients_array @ x_value + intercepts_array))

    def subgradient(x: ObjectiveValue) -> FloatArray:
        x_value = _as_vector(x, dimension=dimension)
        values = gradients_array @ x_value + intercepts_array
        active_index = int(np.argmax(values))
        return gradients_array[active_index].copy()

    # Candidate minimizers for bounded max-affine objectives occur at active-piece
    # intersections. We solve all square systems of `dimension + 1` equations
    # consisting of `dimension` pairwise-equality constraints and one active value.
    candidate_points: list[FloatArray] = []
    num_pieces = gradients_array.shape[0]
    for anchor_index in range(num_pieces):
        other_indices = [index for index in range(num_pieces) if index != anchor_index]
        if len(other_indices) < dimension:
            continue
        from itertools import combinations

        for combo in combinations(other_indices, dimension):
            matrix_rows = []
            rhs = []
            for other_index in combo:
                matrix_rows.append(gradients_array[anchor_index] - gradients_array[other_index])
                rhs.append(intercepts_array[other_index] - intercepts_array[anchor_index])
            matrix = np.asarray(matrix_rows, dtype=float)
            rhs_array = np.asarray(rhs, dtype=float)
            if np.linalg.matrix_rank(matrix) < dimension:
                continue
            candidate_points.append(np.linalg.solve(matrix, rhs_array))

    if not candidate_points:
        raise ValueError("No candidate minimizer could be derived for max-affine objective.")

    minimizer = min(candidate_points, key=objective)

    return ObjectiveDefinition(
        id=objective_id,
        family="max_affine",
        name=f"{dimension}D max-affine objective",
        params={
            "gradients": [
                _json_list(row) for row in gradients_array.tolist()
            ],
            "intercepts": _json_list(intercepts_array),
        },
        dimension=dimension,
        lipschitz_constant=float(np.max(np.linalg.norm(gradients_array, axis=1))),
        minimum_value=float(objective(minimizer)),
        minimizer=minimizer.copy(),
        objective=objective,
        subgradient=subgradient,
    )


def _solve_simplex_max_affine(
    gradients_array: FloatArray,
    intercepts_array: FloatArray,
) -> tuple[FloatArray, float]:
    """Solve `min max_i <a_i, x> + b_i` over the probability simplex."""
    from scipy.optimize import linprog

    dimension = gradients_array.shape[1]
    objective = np.zeros(dimension + 1, dtype=float)
    objective[-1] = 1.0

    lhs = np.column_stack([gradients_array, -np.ones(gradients_array.shape[0])])
    rhs = -intercepts_array
    equality_lhs = np.zeros((1, dimension + 1), dtype=float)
    equality_lhs[0, :dimension] = 1.0
    equality_rhs = np.asarray([1.0], dtype=float)
    bounds = [(0.0, None)] * dimension + [(None, None)]

    result = linprog(
        objective,
        A_ub=lhs,
        b_ub=rhs,
        A_eq=equality_lhs,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Unable to solve simplex max-affine LP: {result.message}")

    minimizer = np.asarray(result.x[:dimension], dtype=float)
    minimum_value = float(result.x[-1])
    return minimizer, minimum_value


def _make_simplex_linear_objective(
    objective_id: str,
    *,
    coefficients: Sequence[float],
) -> ObjectiveDefinition:
    coefficients_array = _to_float_array(coefficients)
    dimension = len(coefficients_array)
    minimizer = np.zeros(dimension, dtype=float)
    minimizer[int(np.argmin(coefficients_array))] = 1.0

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_vector(x, dimension=dimension)
        return float(coefficients_array @ x_value)

    def subgradient(x: ObjectiveValue) -> FloatArray:
        _as_vector(x, dimension=dimension)
        return coefficients_array.copy()

    return ObjectiveDefinition(
        id=objective_id,
        family="simplex_linear",
        name=f"{dimension}D simplex linear objective",
        params={"coefficients": _json_list(coefficients_array)},
        dimension=dimension,
        lipschitz_constant=float(np.max(np.abs(coefficients_array))),
        minimum_value=float(objective(minimizer)),
        minimizer=minimizer.copy(),
        objective=objective,
        subgradient=subgradient,
    )


def _make_simplex_max_affine_objective(
    objective_id: str,
    *,
    gradients: Sequence[Sequence[float]],
    intercepts: Sequence[float],
) -> ObjectiveDefinition:
    gradients_array = np.asarray(gradients, dtype=float)
    intercepts_array = _to_float_array(intercepts)

    if gradients_array.ndim != 2:
        raise ValueError("gradients must define a matrix.")
    if gradients_array.shape[0] != len(intercepts_array):
        raise ValueError("Need one intercept per affine piece.")

    dimension = gradients_array.shape[1]
    minimizer, minimum_value = _solve_simplex_max_affine(gradients_array, intercepts_array)

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_vector(x, dimension=dimension)
        return float(np.max(gradients_array @ x_value + intercepts_array))

    def subgradient(x: ObjectiveValue) -> FloatArray:
        x_value = _as_vector(x, dimension=dimension)
        values = gradients_array @ x_value + intercepts_array
        active_index = int(np.argmax(values))
        return gradients_array[active_index].copy()

    return ObjectiveDefinition(
        id=objective_id,
        family="simplex_max_affine",
        name=f"{dimension}D simplex max-affine objective",
        params={
            "gradients": [_json_list(row) for row in gradients_array.tolist()],
            "intercepts": _json_list(intercepts_array),
        },
        dimension=dimension,
        lipschitz_constant=float(np.max(np.max(np.abs(gradients_array), axis=1))),
        minimum_value=minimum_value,
        minimizer=minimizer.copy(),
        objective=objective,
        subgradient=subgradient,
    )


def _make_boxed_max_affine_from_scales(
    objective_id: str,
    *,
    scales: Sequence[float],
) -> ObjectiveDefinition:
    scales_array = _to_float_array(scales)
    dimension = len(scales_array)
    minimizer = np.zeros(dimension, dtype=float)

    def objective(x: ObjectiveValue) -> float:
        x_value = _as_vector(x, dimension=dimension)
        return float(np.max(scales_array * np.abs(x_value)))

    def subgradient(x: ObjectiveValue) -> FloatArray:
        x_value = _as_vector(x, dimension=dimension)
        values = scales_array * np.abs(x_value)
        active_index = int(np.argmax(values))
        gradient = np.zeros(dimension, dtype=float)
        gradient[active_index] = scales_array[active_index] * _sign_with_zero(
            float(x_value[active_index])
        )
        return gradient

    return ObjectiveDefinition(
        id=objective_id,
        family="ill_conditioned_max_affine",
        name=f"{dimension}D ill-conditioned max-affine objective",
        params={"scales": _json_list(scales_array)},
        dimension=dimension,
        lipschitz_constant=float(np.max(scales_array)),
        minimum_value=0.0,
        minimizer=minimizer.copy(),
        objective=objective,
        subgradient=subgradient,
    )


OBJECTIVE_REGISTRY: Dict[str, ObjectiveDefinition] = {
    "abs_a2": _make_abs_shift_objective("abs_a2", a=2.0),
    "weighted_l1_shift_1d": _make_weighted_l1_shift_1d(
        "weighted_l1_shift_1d",
        weights=[1.25, 0.75, 0.5],
        shifts=[2.0, -1.5, 3.5],
    ),
    "max_affine_1d": _make_max_affine_1d(
        "max_affine_1d",
        slopes=[1.2, -0.8, 0.35],
        intercepts=[-2.0, 2.6, 0.9],
    ),
    "weighted_l1_shift_2d": _make_weighted_l1_shift_nd(
        "weighted_l1_shift_2d",
        shifts=[2.0, -1.0],
        weights=[1.0, 1.5],
    ),
    "linf_shift_2d": _make_linf_shift_nd(
        "linf_shift_2d",
        shifts=[1.5, -2.5],
    ),
    "weighted_l1_shift_4d": _make_weighted_l1_shift_nd(
        "weighted_l1_shift_4d",
        shifts=[2.5, -1.5, 1.0, 3.0],
        weights=[1.0, 0.5, 1.25, 0.75],
    ),
    "max_affine_4d": _make_max_affine_nd(
        "max_affine_4d",
        gradients=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        intercepts=[-2.0, 1.5, -0.75, -2.5, 3.75],
    ),
    "ill_conditioned_l1_shift_8d": _make_weighted_l1_shift_nd(
        "ill_conditioned_l1_shift_8d",
        shifts=[2.0, -1.5, 1.0, 3.0, -2.0, 0.5, 1.5, -0.75],
        weights=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
    ),
    "ill_conditioned_max_affine_8d": _make_boxed_max_affine_from_scales(
        "ill_conditioned_max_affine_8d",
        scales=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
    ),
    "simplex_linear_8d": _make_simplex_linear_objective(
        "simplex_linear_8d",
        coefficients=[0.8, -0.4, 1.2, 0.1, -0.9, 0.3, 0.6, -0.2],
    ),
    "simplex_max_affine_8d": _make_simplex_max_affine_objective(
        "simplex_max_affine_8d",
        gradients=[
            [1.0, -0.5, 0.2, 0.0, -0.3, 0.4, 0.1, -0.2],
            [-0.4, 0.8, -0.1, 0.3, 0.2, -0.5, 0.6, 0.0],
            [0.2, 0.1, 0.9, -0.4, 0.5, 0.0, -0.3, 0.4],
            [0.0, -0.2, 0.4, 1.0, -0.5, 0.3, 0.2, -0.1],
            [-0.3, 0.4, 0.0, -0.2, 0.9, -0.1, 0.5, 0.2],
            [0.5, 0.0, -0.4, 0.2, 0.1, 0.8, -0.2, 0.3],
        ],
        intercepts=[0.0, 0.1, -0.05, 0.2, -0.1, 0.05],
    ),
    "simplex_sparse_max_affine_16d": _make_simplex_max_affine_objective(
        "simplex_sparse_max_affine_16d",
        gradients=[
            [1.0 if j == i else (-0.2 if j == (i + 1) % 16 else 0.0) for j in range(16)]
            for i in range(8)
        ]
        + [
            [-0.5 if j == i else (0.7 if j == (i + 4) % 16 else 0.0) for j in range(16)]
            for i in range(8)
        ],
        intercepts=[0.02 * i for i in range(16)],
    ),
}


def get_objective(objective_id: str) -> ObjectiveDefinition:
    """Return a named objective preset or raise a clear error."""
    try:
        return OBJECTIVE_REGISTRY[objective_id]
    except KeyError as exc:
        available = ", ".join(sorted(OBJECTIVE_REGISTRY))
        raise ValueError(
            f"Unknown objective '{objective_id}'. Available values: {available}."
        ) from exc


def list_objective_ids() -> list[str]:
    """Return available objective IDs in a stable order."""
    return sorted(OBJECTIVE_REGISTRY)
