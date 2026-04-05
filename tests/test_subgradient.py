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

from pda import SubgradientMethod, get_objective


def _zero_prox_center(objective_id: str) -> float | np.ndarray:
    objective = get_objective(objective_id)
    if objective.dimension == 1:
        return 0.0
    return np.zeros(objective.dimension, dtype=float)


class SubgradientShapeTest(unittest.TestCase):
    def test_solver_runs_for_scalar_objective(self) -> None:
        self._assert_solver_shapes("weighted_l1_shift_1d", D=2.0)

    def test_solver_runs_for_two_dimensional_objective(self) -> None:
        self._assert_solver_shapes("linf_shift_2d", D=4.5)

    def test_solver_runs_for_four_dimensional_objective(self) -> None:
        self._assert_solver_shapes("max_affine_4d", D=6.0)

    def _assert_solver_shapes(self, objective_id: str, *, D: float) -> None:
        objective = get_objective(objective_id)
        prox_center = _zero_prox_center(objective_id)
        result = SubgradientMethod(prox_center=prox_center).run(
            gamma=0.75,
            D=D,
            subgradient_oracle=objective.subgradient,
            max_iter=5,
            restrict_to_fd=True,
        )

        self.assertEqual(result["iterations"], 5)
        self.assertEqual(len(result["x"]), result["iterations"] + 1)
        self.assertEqual(len(result["x_hat"]), result["iterations"] + 1)
        self.assertEqual(len(result["g"]), result["iterations"])
        self.assertEqual(len(result["alpha"]), result["iterations"])
        self.assertTrue(all(alpha_k > 0.0 for alpha_k in result["alpha"]))
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

        radius = np.sqrt(2.0 * D)
        for x_k in result["x"]:
            offset = np.asarray(x_k, dtype=float) - np.asarray(prox_center, dtype=float)
            self.assertLessEqual(float(np.linalg.norm(offset)), radius + 1e-10)

        if objective.dimension == 1:
            self.assertIsInstance(result["x"][0], float)
            self.assertIsInstance(result["g"][0], float)
        else:
            self.assertEqual(np.asarray(result["x"][0]).shape, (objective.dimension,))
            self.assertEqual(np.asarray(result["g"][0]).shape, (objective.dimension,))


if __name__ == "__main__":
    unittest.main()
