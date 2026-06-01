from __future__ import annotations

from time import perf_counter
from typing import Callable, List, Literal, TypedDict, Union

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
SolverValue = Union[float, FloatArray]
SubgradientOracle = Callable[[SolverValue], SolverValue]
ProxFunctionName = Literal["euclidean", "euclidian", "weighted_euclidean", "entropy"]
DualAveragingMode = Literal["simple", "weighted"]


class SDAResult(TypedDict):
    """State returned by `SDA.run`, including per-run timing metrics."""

    converged: bool
    iterations: int
    total_runtime_seconds: float
    avg_iteration_time_seconds: float
    restrict_to_fd: bool
    prox_fun: str
    prox_weights: List[float] | None
    dual_averaging: str
    S: List[float]
    x: List[SolverValue]
    x_hat: List[SolverValue]
    s: List[SolverValue]
    g: List[SolverValue]
    B_hat: List[float]
    B: List[float]
    gap_accum: List[float]
    gap: List[float]


class SDA:
    """Dual Averaging solver with closed-form Euclidean and entropy prox-functions.

    The Euclidean prox follows the support-form gap algorithm in the unconstrained setting:

    - `d(x) = 0.5 * ||x - x_0||_2^2`
    - `xi_D(s) = sqrt(2 * D) * ||s||_2`
    - `x_{k+1} = x_0 - s_{k+1} / beta_{k+1}`

    The weighted Euclidean prox uses positive coordinate weights `h_i`:

    - `d(x) = 0.5 * sum_i h_i * (x_i - x_0_i)^2`
    - `x_{k+1,i} = x_0_i - s_{k+1,i} / (beta_{k+1} h_i)`

    The entropy prox is the simplex setup from Nesterov:

    - `Q = {x >= 0 : sum(x) = 1}`
    - `d(x) = log(n) + sum_i x_i log(x_i)`
    - `x_0 = (1 / n, ..., 1 / n)`
    - `pi_beta(s) = softmax(s / beta)`
    """

    def __init__(
        self,
        prox_center: SolverValue = 0,
        prox_fun: ProxFunctionName = "euclidean",
        prox_weights: SolverValue | None = None,
    ) -> None:
        """Initialize the solver state shared by all SDA runs.

        Args:
            prox_center: Prox-center `x_0` used by the prox-function. Entropy prox
                requires the uniform simplex center.
            prox_fun: Name of the prox-function. `"euclidian"` is accepted as a
                backwards-compatible alias for `"euclidean"`.
            prox_weights: Positive coordinate weights for `"weighted_euclidean"`.

        Raises:
            ValueError: If the prox-function or prox-center is invalid.
        """
        if prox_fun == "euclidian":
            prox_fun = "euclidean"
        if prox_fun not in {"euclidean", "weighted_euclidean", "entropy"}:
            raise ValueError(f"Proximal function '{prox_fun}' is not supported.")
        self.prox_center: FloatArray = self._as_array(prox_center)
        self.prox_fun = prox_fun
        self.prox_weights: FloatArray | None = (
            self._as_array(prox_weights) if prox_weights is not None else None
        )
        self.sigma = 1.0
        self._validate_prox_center()

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

    def _validate_prox_center(self) -> None:
        """Validate prox-center assumptions for the selected prox-function."""
        if self.prox_fun == "weighted_euclidean":
            if self.prox_weights is None:
                raise ValueError("weighted_euclidean prox requires prox_weights.")
            if self.prox_weights.shape != self.prox_center.shape:
                raise ValueError(
                    "prox_weights must have the same shape as prox_center for weighted_euclidean prox."
                )
            if np.any(self.prox_weights <= 0.0):
                raise ValueError("prox_weights must be positive for weighted_euclidean prox.")
        if self.prox_fun == "entropy":
            if self.prox_center.ndim != 1 or self.prox_center.size < 2:
                raise ValueError("Entropy prox requires a vector prox_center.")
            expected = np.full(self.prox_center.shape, 1.0 / self.prox_center.size)
            if not np.allclose(self.prox_center, expected):
                raise ValueError("Entropy prox requires the uniform simplex prox_center.")

    def _validate_shape(self, value: FloatArray, *, name: str) -> None:
        """Ensure vector/scalar iterates match the prox-center shape."""
        if value.shape != self.prox_center.shape:
            raise ValueError(
                f"{name} has shape {value.shape}, expected {self.prox_center.shape}."
            )

    @staticmethod
    def _entropy_value(x_array: FloatArray) -> float:
        """Evaluate `log(n) + sum_i x_i log(x_i)` on the simplex."""
        positive = x_array > 0.0
        return float(np.log(x_array.size) + np.sum(x_array[positive] * np.log(x_array[positive])))

    @staticmethod
    def _softmax(values: FloatArray) -> FloatArray:
        """Return a numerically stable softmax."""
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    def _prox_fun(self, x: SolverValue) -> float:
        """Evaluate the selected prox-function `d(x)`."""
        x_array = self._as_array(x)
        self._validate_shape(x_array, name="x")
        if self.prox_fun == "euclidean":
            return float(0.5 * np.sum((x_array - self.prox_center) ** 2))
        if self.prox_fun == "weighted_euclidean":
            assert self.prox_weights is not None
            return float(0.5 * np.sum(self.prox_weights * (x_array - self.prox_center) ** 2))

        if np.any(x_array < -1e-12) or not np.isclose(np.sum(x_array), 1.0):
            raise ValueError("Entropy prox is defined only on the probability simplex.")
        clipped = np.clip(x_array, 0.0, None)
        clipped = clipped / np.sum(clipped)
        return self._entropy_value(clipped)

    def _xi(self, D: float, s: SolverValue) -> float:
        """Evaluate the support term `xi_D(s)` for the selected prox."""
        s_array = self._as_array(s)
        self._validate_shape(s_array, name="s")
        if self.prox_fun == "euclidean":
            return float(np.sqrt(2.0 * D * np.sum(s_array ** 2)))
        if self.prox_fun == "weighted_euclidean":
            assert self.prox_weights is not None
            return float(np.sqrt(2.0 * D * np.sum((s_array ** 2) / self.prox_weights)))

        return self._entropy_xi(D, s_array)

    def _primal_iterate(self, beta: float, s: SolverValue) -> FloatArray:
        """Compute the primal update `pi_beta(s)` in closed form."""
        s_array = self._as_array(s)
        self._validate_shape(s_array, name="s")
        if self.prox_fun == "euclidean":
            return self.prox_center + s_array / beta
        if self.prox_fun == "weighted_euclidean":
            assert self.prox_weights is not None
            return self.prox_center + s_array / (beta * self.prox_weights)
        return self._softmax(s_array / beta)

    def _project_to_fd(self, D: float, x: SolverValue) -> FloatArray:
        """Map a point into the restriction set `F_D = {x : d(x) <= D}`."""
        x_array = self._as_array(x)
        self._validate_shape(x_array, name="x")
        if self.prox_fun == "euclidean":
            offset = x_array - self.prox_center
            radius = np.sqrt(2.0 * D)
            norm = float(np.linalg.norm(offset))

            if norm <= radius or norm == 0.0:
                return x_array
            return self.prox_center + (radius / norm) * offset
        if self.prox_fun == "weighted_euclidean":
            assert self.prox_weights is not None
            offset = x_array - self.prox_center
            radius = np.sqrt(2.0 * D)
            norm = float(np.sqrt(np.sum(self.prox_weights * offset ** 2)))

            if norm <= radius or norm == 0.0:
                return x_array
            return self.prox_center + (radius / norm) * offset

        simplex_x = np.clip(x_array, 0.0, None)
        simplex_sum = float(np.sum(simplex_x))
        if simplex_sum == 0.0:
            return self.prox_center.copy()
        simplex_x = simplex_x / simplex_sum
        if self._prox_fun(simplex_x) <= D:
            return simplex_x
        if D == 0.0:
            return self.prox_center.copy()

        low = 0.0
        high = 1.0
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            candidate = self.prox_center + midpoint * (simplex_x - self.prox_center)
            if self._prox_fun(candidate) <= D:
                low = midpoint
            else:
                high = midpoint
        return self.prox_center + low * (simplex_x - self.prox_center)

    def _dual_norm(self, value: SolverValue) -> float:
        """Return the dual norm associated with the selected prox geometry."""
        value_array = self._as_array(value)
        self._validate_shape(value_array, name="value")
        if self.prox_fun == "euclidean":
            return float(np.linalg.norm(value_array))
        if self.prox_fun == "weighted_euclidean":
            assert self.prox_weights is not None
            return float(np.sqrt(np.sum((value_array ** 2) / self.prox_weights)))
        return float(np.max(np.abs(value_array)))

    def _entropy_xi(self, D: float, s_array: FloatArray) -> float:
        """Evaluate entropy-prox support over `{x in simplex : d(x) <= D}`."""
        if D <= 0.0 or np.allclose(s_array, s_array[0]):
            return 0.0

        max_entropy_distance = float(np.log(s_array.size))
        centered_mean = float(np.mean(s_array))
        if D >= max_entropy_distance:
            return float(np.max(s_array) - centered_mean)

        def entropy_at_temperature(temperature: float) -> tuple[float, FloatArray]:
            weights = self._softmax(s_array / temperature)
            return self._entropy_value(weights), weights

        scale = max(1.0, float(np.max(np.abs(s_array))))
        low = 1e-12 * scale
        high = 1.0
        high_entropy, _ = entropy_at_temperature(high)
        while high_entropy > D:
            high *= 2.0
            high_entropy, _ = entropy_at_temperature(high)

        weights = self.prox_center
        for _ in range(120):
            midpoint = 0.5 * (low + high)
            midpoint_entropy, midpoint_weights = entropy_at_temperature(midpoint)
            if midpoint_entropy > D:
                low = midpoint
            else:
                high = midpoint
                weights = midpoint_weights
        return float(np.dot(s_array, weights - self.prox_center))

    def run(
        self,
        gamma: float,
        D: float,
        eps: float,
        subgradient_oracle: SubgradientOracle,
        max_iter: int,
        restrict_to_fd: bool = False,
        dual_averaging: DualAveragingMode = "simple",
    ) -> SDAResult:
        """Run dual averaging and return the tracked trajectories.

        Args:
            gamma: Positive SDA scaling parameter.
            D: Nonnegative radius of the restricted set `F_D = {x : d(x) <= D}`.
            eps: Nonnegative stopping tolerance for `delta_k(D) / (k + 1)`.
            subgradient_oracle: Callable returning a subgradient at the current
                iterate. It should accept the same scalar/vector shape returned
                by the solver.
            max_iter: Maximum number of iterations to execute.
            restrict_to_fd: Whether to project each primal iterate onto `F_D`.
            dual_averaging: `"simple"` uses `lambda_k = 1`; `"weighted"` uses
                `lambda_k = 1 / ||g_k||_*`, as in Nesterov's weighted dual
                averages method.

        Returns:
            A dictionary containing the SDA trajectories. The keys follow the
            original variable names used in the implementation: `x`, `x_hat`,
            `s`, `g`, `B_hat`, `B`, `gap_accum`, and `gap`. The result also
            includes `converged`, `iterations`, `total_runtime_seconds`,
            `avg_iteration_time_seconds`, and `restrict_to_fd`.

        Raises:
            ValueError: If the numeric parameters are invalid.
        """
        self._validate_inputs(
            gamma=gamma,
            D=D,
            eps=eps,
            max_iter=max_iter,
            dual_averaging=dual_averaging,
        )
        start_time = perf_counter()

        x: List[FloatArray] = [self.prox_center.copy()]
        x_hat: List[FloatArray] = [self.prox_center.copy()]
        s: List[FloatArray] = [np.zeros_like(self.prox_center, dtype=float)]
        g: List[FloatArray] = []

        S: List[float] = [0.0]
        B_hat: List[float] = [1.0]
        B: List[float] = [0.0]
        gap_accum: List[float] = [0.0]
        gap: List[float] = []

        for k in range(max_iter):
            oracle_input = self._to_public_value(x[k])
            g_k = self._as_array(subgradient_oracle(oracle_input))
            self._validate_shape(g_k, name="subgradient_oracle result")
            lambda_k = self._dual_average_weight(g_k, dual_averaging)

            g.append(g_k)
            s.append(s[k] + lambda_k * g_k)
            S.append(S[k] + lambda_k)

            if k == 0:
                B_hat.append(1.0)
            else:
                B_hat.append(B_hat[k] + 1.0 / B_hat[k])

            B.append(gamma * B_hat[k + 1])
            x_next = self._primal_iterate(B[k + 1], -s[k + 1])
            if restrict_to_fd:
                x_next = self._project_to_fd(D, x_next)
            x.append(x_next)

            if S[k + 1] > 0.0:
                x_hat.append((S[k] * x_hat[k] + lambda_k * x[k]) / S[k + 1])
            else:
                x_hat.append(x_hat[k].copy())

            gap_accum.append(
                gap_accum[k] + lambda_k * float(np.sum(g_k * (x[k] - self.prox_center)))
            )
            gap.append(gap_accum[k + 1] + self._xi(D, -s[k + 1]))

            denominator = S[k + 1] if S[k + 1] > 0.0 else 1.0
            if gap[k] / denominator <= eps:
                total_runtime_seconds = perf_counter() - start_time
                iterations = k + 1
                return self._build_result(
                    converged=True,
                    iterations=iterations,
                    total_runtime_seconds=total_runtime_seconds,
                    avg_iteration_time_seconds=total_runtime_seconds / iterations,
                    restrict_to_fd=restrict_to_fd,
                    dual_averaging=dual_averaging,
                    S=S,
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
            dual_averaging=dual_averaging,
            S=S,
            x=x,
            x_hat=x_hat,
            s=s,
            g=g,
            B_hat=B_hat,
            B=B,
            gap_accum=gap_accum,
            gap=gap,
        )

    def _dual_average_weight(
        self,
        g_k: FloatArray,
        dual_averaging: DualAveragingMode,
    ) -> float:
        """Return `lambda_k` for the selected dual averaging mode."""
        if dual_averaging == "simple":
            return 1.0

        dual_norm = self._dual_norm(g_k)
        if dual_norm == 0.0:
            return 0.0
        return 1.0 / dual_norm

    @staticmethod
    def _validate_inputs(
        gamma: float,
        D: float,
        eps: float,
        max_iter: int,
        dual_averaging: str,
    ) -> None:
        """Validate numeric solver parameters."""
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if D < 0:
            raise ValueError("D must be nonnegative.")
        if eps < 0:
            raise ValueError("eps must be nonnegative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if dual_averaging not in {"simple", "weighted"}:
            raise ValueError("dual_averaging must be either 'simple' or 'weighted'.")

    def _build_result(
        self,
        converged: bool,
        iterations: int,
        total_runtime_seconds: float,
        avg_iteration_time_seconds: float,
        restrict_to_fd: bool,
        dual_averaging: str,
        S: List[float],
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
            "prox_fun": self.prox_fun,
            "prox_weights": (
                self.prox_weights.astype(float).tolist()
                if self.prox_weights is not None
                else None
            ),
            "dual_averaging": dual_averaging,
            "S": S,
            "x": [self._to_public_value(value) for value in x],
            "x_hat": [self._to_public_value(value) for value in x_hat],
            "s": [self._to_public_value(value) for value in s],
            "g": [self._to_public_value(value) for value in g],
            "B_hat": B_hat,
            "B": B,
            "gap_accum": gap_accum,
            "gap": gap,
        }


class WeightedDualAveraging(SDA):
    """Convenience wrapper for Nesterov's weighted dual averages method."""

    def run(
        self,
        gamma: float,
        D: float,
        eps: float,
        subgradient_oracle: SubgradientOracle,
        max_iter: int,
        restrict_to_fd: bool = False,
        dual_averaging: DualAveragingMode = "weighted",
    ) -> SDAResult:
        """Run weighted dual averaging with `lambda_k = 1 / ||g_k||_*`."""
        if dual_averaging != "weighted":
            raise ValueError("WeightedDualAveraging always uses dual_averaging='weighted'.")
        return super().run(
            gamma=gamma,
            D=D,
            eps=eps,
            subgradient_oracle=subgradient_oracle,
            max_iter=max_iter,
            restrict_to_fd=restrict_to_fd,
            dual_averaging="weighted",
        )
