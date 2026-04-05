from __future__ import annotations

from time import perf_counter
from typing import Callable, List, TypedDict, Union

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
SolverValue = Union[float, FloatArray]
SubgradientOracle = Callable[[SolverValue], SolverValue]


class SubgradientResult(TypedDict):
    """State returned by `SubgradientMethod.run`, including timing metrics."""

    iterations: int
    total_runtime_seconds: float
    avg_iteration_time_seconds: float
    restrict_to_fd: bool
    x: List[SolverValue]
    x_hat: List[SolverValue]
    g: List[SolverValue]
    alpha: List[float]


class SubgradientMethod:
    """Projected subgradient method for the Euclidean prox-function."""

    def __init__(self, prox_center: SolverValue = 0, prox_fun: str = "euclidian") -> None:
        """Initialize the solver state shared by all subgradient runs."""
        if prox_fun != "euclidian":
            raise ValueError(f"Proximal function '{prox_fun}' is not supported.")

        self.prox_center: FloatArray = self._as_array(prox_center)
        self.prox_fun = prox_fun

    @staticmethod
    def _as_array(value: SolverValue) -> FloatArray:
        """Convert a scalar or vector-like value to a float NumPy array."""
        return np.asarray(value, dtype=float)

    @staticmethod
    def _to_public_value(value: FloatArray) -> SolverValue:
        """Return scalars as `float` and vectors as NumPy arrays."""
        if value.ndim == 0:
            return float(value)
        return value

    def _project_to_fd(self, D: float, x: SolverValue) -> FloatArray:
        """Project a point onto the Euclidean restriction set `F_D`."""
        x_array = self._as_array(x)
        offset = x_array - self.prox_center
        radius = np.sqrt(2.0 * D)
        norm = float(np.linalg.norm(offset))

        if norm <= radius or norm == 0.0:
            return x_array
        return self.prox_center + (radius / norm) * offset

    def run(
        self,
        gamma: float,
        D: float,
        subgradient_oracle: SubgradientOracle,
        max_iter: int,
        restrict_to_fd: bool = False,
    ) -> SubgradientResult:
        """Run the projected subgradient method and return tracked trajectories.

        The returned result includes `iterations`, `total_runtime_seconds`,
        `avg_iteration_time_seconds`, `restrict_to_fd`, and the tracked
        trajectories.
        """
        self._validate_inputs(gamma=gamma, D=D, max_iter=max_iter)
        start_time = perf_counter()

        x: List[FloatArray] = [self.prox_center.copy()]
        x_hat: List[FloatArray] = [self.prox_center.copy()]
        g: List[FloatArray] = []
        alpha: List[float] = []

        for k in range(max_iter):
            oracle_input = self._to_public_value(x[k])
            g_k = self._as_array(subgradient_oracle(oracle_input))
            if g_k.shape != self.prox_center.shape:
                raise ValueError(
                    "subgradient_oracle returned a value with shape "
                    f"{g_k.shape}, expected {self.prox_center.shape}."
                )

            alpha_k = float(gamma / np.sqrt(k + 1.0))
            x_next = x[k] - alpha_k * g_k
            if restrict_to_fd:
                x_next = self._project_to_fd(D, x_next)

            g.append(g_k)
            alpha.append(alpha_k)
            x.append(x_next)
            x_hat.append((k * x_hat[k] + x[k]) / (k + 1))

        total_runtime_seconds = perf_counter() - start_time
        return self._build_result(
            iterations=max_iter,
            total_runtime_seconds=total_runtime_seconds,
            avg_iteration_time_seconds=total_runtime_seconds / max_iter,
            restrict_to_fd=restrict_to_fd,
            x=x,
            x_hat=x_hat,
            g=g,
            alpha=alpha,
        )

    @staticmethod
    def _validate_inputs(gamma: float, D: float, max_iter: int) -> None:
        """Validate numeric solver parameters."""
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if D < 0:
            raise ValueError("D must be nonnegative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

    def _build_result(
        self,
        iterations: int,
        total_runtime_seconds: float,
        avg_iteration_time_seconds: float,
        restrict_to_fd: bool,
        x: List[FloatArray],
        x_hat: List[FloatArray],
        g: List[FloatArray],
        alpha: List[float],
    ) -> SubgradientResult:
        """Convert internal NumPy state into a stable public result format."""
        return {
            "iterations": iterations,
            "total_runtime_seconds": total_runtime_seconds,
            "avg_iteration_time_seconds": avg_iteration_time_seconds,
            "restrict_to_fd": restrict_to_fd,
            "x": [self._to_public_value(value) for value in x],
            "x_hat": [self._to_public_value(value) for value in x_hat],
            "g": [self._to_public_value(value) for value in g],
            "alpha": alpha,
        }
