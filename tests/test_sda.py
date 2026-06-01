from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pda import SDA, WeightedDualAveraging, get_objective


def _zero_prox_center(objective_id: str) -> float | np.ndarray:
    objective = get_objective(objective_id)
    if objective.dimension == 1:
        return 0.0
    return np.zeros(objective.dimension, dtype=float)


class SDAShapeTest(unittest.TestCase):
    def test_solver_runs_for_scalar_objective(self) -> None:
        self._assert_solver_shapes("weighted_l1_shift_1d", D=2.0)

    def test_solver_runs_for_two_dimensional_objective(self) -> None:
        self._assert_solver_shapes("linf_shift_2d", D=4.0)

    def test_solver_runs_for_four_dimensional_objective(self) -> None:
        self._assert_solver_shapes("max_affine_4d", D=6.0)

    def _assert_solver_shapes(self, objective_id: str, *, D: float) -> None:
        objective = get_objective(objective_id)
        prox_center = _zero_prox_center(objective_id)
        gamma = objective.lipschitz_constant / np.sqrt(2.0 * D)

        result = SDA(prox_center=prox_center).run(
            gamma=gamma,
            D=D,
            eps=1e-3,
            subgradient_oracle=objective.subgradient,
            max_iter=5,
            restrict_to_fd=True,
        )

        self.assertGreaterEqual(result["iterations"], 1)
        self.assertEqual(len(result["x"]), result["iterations"] + 1)
        self.assertEqual(len(result["g"]), result["iterations"])
        self.assertIn("total_runtime_seconds", result)
        self.assertIn("avg_iteration_time_seconds", result)
        self.assertGreaterEqual(result["total_runtime_seconds"], 0.0)
        self.assertGreaterEqual(result["avg_iteration_time_seconds"], 0.0)
        self.assertTrue(
            math.isclose(
                result["avg_iteration_time_seconds"],
                result["total_runtime_seconds"] / result["iterations"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        if objective.dimension == 1:
            self.assertIsInstance(result["x"][0], float)
            self.assertIsInstance(result["g"][0], float)
        else:
            self.assertEqual(np.asarray(result["x"][0]).shape, (objective.dimension,))
            self.assertEqual(np.asarray(result["g"][0]).shape, (objective.dimension,))

    def test_euclidian_alias_keeps_existing_euclidean_update(self) -> None:
        solver = SDA(prox_center=np.zeros(2), prox_fun="euclidian")

        result = solver.run(
            gamma=2.0,
            D=3.0,
            eps=0.0,
            subgradient_oracle=lambda _: np.asarray([2.0, -4.0]),
            max_iter=1,
        )

        self.assertEqual(result["prox_fun"], "euclidean")
        np.testing.assert_allclose(result["s"][1], np.asarray([2.0, -4.0]))
        np.testing.assert_allclose(result["x"][1], np.asarray([-1.0, 2.0]))
        self.assertEqual(result["S"], [0.0, 1.0])

    def test_weighted_dual_averaging_normalizes_by_dual_norm(self) -> None:
        solver = WeightedDualAveraging(prox_center=np.zeros(2))

        result = solver.run(
            gamma=2.0,
            D=3.0,
            eps=0.0,
            subgradient_oracle=lambda _: np.asarray([3.0, 4.0]),
            max_iter=1,
        )

        self.assertEqual(result["dual_averaging"], "weighted")
        self.assertEqual(result["S"], [0.0, 0.2])
        np.testing.assert_allclose(result["s"][1], np.asarray([0.6, 0.8]))
        np.testing.assert_allclose(result["x"][1], np.asarray([-0.3, -0.4]))

    def test_entropy_prox_uses_simplex_softmax_and_linf_dual_norm(self) -> None:
        prox_center = np.full(3, 1.0 / 3.0)
        solver = SDA(prox_center=prox_center, prox_fun="entropy")

        result = solver.run(
            gamma=2.0,
            D=np.log(3.0),
            eps=0.0,
            subgradient_oracle=lambda _: np.asarray([2.0, -1.0, 0.0]),
            max_iter=1,
            dual_averaging="weighted",
        )

        self.assertEqual(result["prox_fun"], "entropy")
        self.assertEqual(result["dual_averaging"], "weighted")
        self.assertEqual(result["S"], [0.0, 0.5])
        np.testing.assert_allclose(result["s"][1], np.asarray([1.0, -0.5, 0.0]))
        self.assertAlmostEqual(float(np.sum(result["x"][1])), 1.0, places=12)
        self.assertTrue(np.all(np.asarray(result["x"][1]) >= 0.0))
        np.testing.assert_allclose(
            result["x"][1],
            np.exp(np.asarray([-0.5, 0.25, 0.0]))
            / np.sum(np.exp(np.asarray([-0.5, 0.25, 0.0]))),
        )

    def test_weighted_euclidean_prox_formulas(self) -> None:
        prox_center = np.asarray([1.0, -1.0])
        prox_weights = np.asarray([2.0, 0.5])
        solver = SDA(
            prox_center=prox_center,
            prox_fun="weighted_euclidean",
            prox_weights=prox_weights,
        )

        x = np.asarray([3.0, 1.0])
        s = np.asarray([4.0, 2.0])
        self.assertAlmostEqual(solver._prox_fun(x), 5.0)
        self.assertAlmostEqual(solver._xi(2.0, s), 8.0)
        self.assertAlmostEqual(solver._dual_norm(s), np.sqrt(16.0 / 2.0 + 4.0 / 0.5))
        np.testing.assert_allclose(
            solver._primal_iterate(2.0, s),
            np.asarray([2.0, 1.0]),
        )
        projected = solver._project_to_fd(0.5, x)
        self.assertLessEqual(solver._prox_fun(projected), 0.5 + 1e-12)

    def test_entropy_restrict_to_fd_returns_point_inside_entropy_ball(self) -> None:
        prox_center = np.full(3, 1.0 / 3.0)
        solver = SDA(prox_center=prox_center, prox_fun="entropy")
        D = 0.05

        result = solver.run(
            gamma=0.2,
            D=D,
            eps=0.0,
            subgradient_oracle=lambda _: np.asarray([-4.0, 2.0, 2.0]),
            max_iter=1,
            restrict_to_fd=True,
        )

        self.assertLessEqual(solver._prox_fun(result["x"][1]), D + 1e-12)

    def test_entropy_xi_respects_entropy_radius(self) -> None:
        prox_center = np.full(3, 1.0 / 3.0)
        solver = SDA(prox_center=prox_center, prox_fun="entropy")
        s = np.asarray([2.0, 0.0, -1.0])

        self.assertEqual(solver._xi(0.0, s), 0.0)
        self.assertGreater(solver._xi(0.1, s), 0.0)
        self.assertLess(solver._xi(0.1, s), np.max(s) - np.mean(s))
        self.assertAlmostEqual(
            solver._xi(np.log(3.0), s),
            np.max(s) - np.mean(s),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
