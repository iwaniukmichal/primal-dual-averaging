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

from pda import get_objective, list_objective_ids


SAMPLE_POINTS = {
    "abs_a2": (0.5, 2.0),
    "weighted_l1_shift_1d": (1.25, 2.0),
    "max_affine_1d": (0.75, 2.0),
    "weighted_l1_shift_2d": (np.array([0.5, -0.5]), np.array([2.0, -1.0])),
    "linf_shift_2d": (np.array([0.0, -1.0]), np.array([1.5, -2.5])),
    "weighted_l1_shift_4d": (
        np.array([0.0, -0.5, 0.25, 1.0]),
        np.array([2.5, -1.5, 1.0, 3.0]),
    ),
    "max_affine_4d": (
        np.array([0.5, -0.25, 0.75, 1.0]),
        np.array([2.0, -1.5, 0.75, 2.5]),
    ),
}


class ObjectiveRegistryTest(unittest.TestCase):
    def test_expected_registry_ids_are_present(self) -> None:
        self.assertEqual(
            list_objective_ids(),
            [
                "abs_a2",
                "linf_shift_2d",
                "max_affine_1d",
                "max_affine_4d",
                "weighted_l1_shift_1d",
                "weighted_l1_shift_2d",
                "weighted_l1_shift_4d",
            ],
        )

    def test_objectives_return_finite_values_and_correct_subgradient_shapes(self) -> None:
        for objective_id in list_objective_ids():
            with self.subTest(objective_id=objective_id):
                objective = get_objective(objective_id)
                smooth_point, nonsmooth_point = SAMPLE_POINTS[objective_id]

                for point in (smooth_point, nonsmooth_point):
                    value = objective.objective(point)
                    self.assertIsInstance(value, float)
                    self.assertTrue(math.isfinite(value))

                    subgradient = objective.subgradient(point)
                    if objective.dimension == 1:
                        self.assertIsInstance(subgradient, float)
                        self.assertTrue(math.isfinite(subgradient))
                    else:
                        self.assertEqual(np.asarray(subgradient).shape, (objective.dimension,))
                        self.assertTrue(np.all(np.isfinite(subgradient)))

    def test_registry_metadata_is_internally_consistent(self) -> None:
        for objective_id in list_objective_ids():
            with self.subTest(objective_id=objective_id):
                objective = get_objective(objective_id)
                self.assertGreaterEqual(objective.dimension, 1)
                self.assertGreater(objective.lipschitz_constant, 0.0)
                self.assertIsInstance(dict(objective.params), dict)
                self.assertTrue(math.isfinite(objective.minimum_value))

                if objective.dimension == 1:
                    self.assertIsInstance(objective.minimizer, float)
                else:
                    self.assertEqual(np.asarray(objective.minimizer).shape, (objective.dimension,))

                self.assertAlmostEqual(
                    objective.objective(objective.minimizer),
                    objective.minimum_value,
                    places=10,
                )

    def test_bounded_max_affine_4d_has_exact_optimum(self) -> None:
        objective = get_objective("max_affine_4d")
        minimizer = np.asarray(objective.minimizer, dtype=float)

        self.assertAlmostEqual(objective.objective(minimizer), 0.0, places=10)
        self.assertEqual(np.asarray(objective.subgradient(minimizer)).shape, (4,))

        sampled_directions = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 1.0, 0.0]),
            np.array([-1.0, -1.0, -1.0, -1.0]),
        ]
        for direction in sampled_directions:
            direction = direction / np.linalg.norm(direction)
            values = [
                objective.objective(minimizer + scale * direction)
                for scale in (-10.0, -5.0, 0.0, 5.0, 10.0)
            ]
            self.assertGreaterEqual(min(values), -1e-9)


if __name__ == "__main__":
    unittest.main()
