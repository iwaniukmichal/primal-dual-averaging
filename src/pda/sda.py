from __future__ import annotations

from time import perf_counter
from typing import Callable, List, TypedDict, Union

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
SolverValue = Union[float, FloatArray]
SubgradientOracle = Callable[[SolverValue], SolverValue]


class SDAResult(TypedDict):
    """State returned by `SDA.run`, including per-run timing metrics."""

    converged: bool
    iterations: int
    total_runtime_seconds: float
    avg_iteration_time_seconds: float
    restrict_to_fd: bool
    x: List[SolverValue]
    x_hat: List[SolverValue]
    s: List[SolverValue]
    g: List[SolverValue]
    B_hat: List[float]
    B: List[float]
    gap_accum: List[float]
    gap: List[float]


class SDA:
    """Simple Dual Averaging solver for the Euclidean prox-function.

    This implementation follows the support-form gap algorithm in the unconstrained Euclidean setting:

    - `d(x) = 0.5 * ||x - x_0||_2^2`
    - `xi_D(s) = sqrt(2 * D) * ||s||_2`
    - `x_{k+1} = x_0 - s_{k+1} / beta_{k+1}`

    The class intentionally supports only the `"euclidian"` prox-function.
    """

    def __init__(self, prox_center: SolverValue = 0, prox_fun: str = "euclidian") -> None:
        """Initialize the solver state shared by all SDA runs.

        Args:
            prox_center: Prox-center `x_0` used in the Euclidean prox-function.
            prox_fun: Name of the prox-function. Only `"euclidian"` is supported.

        Raises:
            ValueError: If a prox-function other than `"euclidian"` is requested.
        """
        if prox_fun != "euclidian":
            raise ValueError(
                f"Proximal function '{prox_fun}' is not supported.")

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

    def _prox_fun(self, x: SolverValue) -> float:
        """Evaluate the Euclidean prox-function `d(x)`."""
        x_array = self._as_array(x)
        return float(0.5 * np.sum((x_array - self.prox_center) ** 2))

    def _xi(self, D: float, s: SolverValue) -> float:
        """Evaluate the support term `xi_D(s)` for the Euclidean prox."""
        s_array = self._as_array(s)
        return float(np.sqrt(2.0 * D * np.sum(s_array ** 2)))

    def _primal_iterate(self, beta: float, s: SolverValue) -> FloatArray:
        """Compute the Euclidean primal update `pi_beta(s)` in closed form."""
        s_array = self._as_array(s)
        return self.prox_center + s_array / beta

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
        eps: float,
        subgradient_oracle: SubgradientOracle,
        max_iter: int,
        restrict_to_fd: bool = False,
    ) -> SDAResult:
        """Run Simple Dual Averaging and return the tracked trajectories.

        Args:
            gamma: Positive SDA scaling parameter.
            D: Nonnegative radius of the restricted set `F_D = {x : d(x) <= D}`.
            eps: Nonnegative stopping tolerance for `delta_k(D) / (k + 1)`.
            subgradient_oracle: Callable returning a subgradient at the current
                iterate. It should accept the same scalar/vector shape returned
                by the solver.
            max_iter: Maximum number of iterations to execute.
            restrict_to_fd: Whether to project each primal iterate onto `F_D`.

        Returns:
            A dictionary containing the SDA trajectories. The keys follow the
            original variable names used in the implementation: `x`, `x_hat`,
            `s`, `g`, `B_hat`, `B`, `gap_accum`, and `gap`. The result also
            includes `converged`, `iterations`, `total_runtime_seconds`,
            `avg_iteration_time_seconds`, and `restrict_to_fd`.

        Raises:
            ValueError: If the numeric parameters are invalid.
        """
        self._validate_inputs(gamma=gamma, D=D, eps=eps, max_iter=max_iter)
        start_time = perf_counter()

        x: List[FloatArray] = [self.prox_center.copy()]
        x_hat: List[FloatArray] = [self.prox_center.copy()]
        s: List[FloatArray] = [np.zeros_like(self.prox_center, dtype=float)]
        g: List[FloatArray] = []

        B_hat: List[float] = [1.0]
        B: List[float] = [0.0]
        gap_accum: List[float] = [0.0]
        gap: List[float] = []

        for k in range(max_iter):
            oracle_input = self._to_public_value(x[k])
            g_k = self._as_array(subgradient_oracle(oracle_input))

            g.append(g_k)
            s.append(s[k] + g_k)

            if k == 0:
                B_hat.append(1.0)
            else:
                B_hat.append(B_hat[k] + 1.0 / B_hat[k])

            B.append(gamma * B_hat[k + 1])
            x_next = self._primal_iterate(B[k + 1], -s[k + 1])
            if restrict_to_fd:
                x_next = self._project_to_fd(D, x_next)
            x.append(x_next)

            x_hat.append((k * x_hat[k] + x[k]) / (k + 1))

            gap_accum.append(
                gap_accum[k] + float(np.sum(g_k * (x[k] - self.prox_center)))
            )
            gap.append(gap_accum[k + 1] + self._xi(D, -s[k + 1]))

            if gap[k] / (k + 1) <= eps:
                total_runtime_seconds = perf_counter() - start_time
                iterations = k + 1
                return self._build_result(
                    converged=True,
                    iterations=iterations,
                    total_runtime_seconds=total_runtime_seconds,
                    avg_iteration_time_seconds=total_runtime_seconds / iterations,
                    restrict_to_fd=restrict_to_fd,
                    x=x,
                    x_hat=x_hat,
                    s=s,
                    g=g,
                    B_hat=B_hat,
                    B=B,
                    gap_accum=gap_accum,
                    gap=gap,
                )

        total_runtime_seconds = perf_counter() - start_time
        return self._build_result(
            converged=False,
            iterations=max_iter,
            total_runtime_seconds=total_runtime_seconds,
            avg_iteration_time_seconds=total_runtime_seconds / max_iter,
            restrict_to_fd=restrict_to_fd,
            x=x,
            x_hat=x_hat,
            s=s,
            g=g,
            B_hat=B_hat,
            B=B,
            gap_accum=gap_accum,
            gap=gap,
        )

    @staticmethod
    def _validate_inputs(gamma: float, D: float, eps: float, max_iter: int) -> None:
        """Validate numeric solver parameters."""
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if D < 0:
            raise ValueError("D must be nonnegative.")
        if eps < 0:
            raise ValueError("eps must be nonnegative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

    def _build_result(
        self,
        converged: bool,
        iterations: int,
        total_runtime_seconds: float,
        avg_iteration_time_seconds: float,
        restrict_to_fd: bool,
        x: List[FloatArray],
        x_hat: List[FloatArray],
        s: List[FloatArray],
        g: List[FloatArray],
        B_hat: List[float],
        B: List[float],
        gap_accum: List[float],
        gap: List[float],
    ) -> SDAResult:
        """Convert internal NumPy state into a stable public result format."""
        return {
            "converged": converged,
            "iterations": iterations,
            "total_runtime_seconds": total_runtime_seconds,
            "avg_iteration_time_seconds": avg_iteration_time_seconds,
            "restrict_to_fd": restrict_to_fd,
            "x": [self._to_public_value(value) for value in x],
            "x_hat": [self._to_public_value(value) for value in x_hat],
            "s": [self._to_public_value(value) for value in s],
            "g": [self._to_public_value(value) for value in g],
            "B_hat": B_hat,
            "B": B,
            "gap_accum": gap_accum,
            "gap": gap,
        }
