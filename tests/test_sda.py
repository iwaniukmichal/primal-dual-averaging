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

from pda import SDA, get_objective


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


if __name__ == "__main__":
    unittest.main()
