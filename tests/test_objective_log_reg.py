from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pda import build_logistic_regression_objective, load_binary_classification_dataset
from pda.objective_log_reg import train_test_split


class LogisticRegressionObjectiveTest(unittest.TestCase):
    def test_loader_drops_string_columns_and_rows_with_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["city", "target", "x1", "x2", "note"])
                writer.writerow(["Warsaw", 0, 1.0, 2.0, "low"])
                writer.writerow(["Paris", 1, 2.0, "", "missing"])
                writer.writerow(["Rome", 1, 3.0, 4.0, "high"])

            X, y = load_binary_classification_dataset(dataset_path)

            np.testing.assert_allclose(X, np.asarray([[1.0, 2.0], [3.0, 4.0]]))
            np.testing.assert_allclose(y, np.asarray([0.0, 1.0]))

    def test_loader_uses_last_column_as_label_when_no_standard_name_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "fallback.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["x2", "comment", "x1", "outcome"])
                writer.writerow([2.0, "alpha", 1.0, 0])
                writer.writerow([4.0, "beta", 3.0, 1])

            X, y = load_binary_classification_dataset(dataset_path)

            np.testing.assert_allclose(X, np.asarray([[2.0, 1.0], [4.0, 3.0]]))
            np.testing.assert_allclose(y, np.asarray([0.0, 1.0]))

    def test_builder_standardizes_features_uses_stratified_split_and_lasso_terms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "synthetic.csv"
            self._write_dataset(dataset_path)

            plain = build_logistic_regression_objective(dataset_path, seed=3)
            lasso = build_logistic_regression_objective(
                dataset_path,
                seed=3,
                lasso=True,
                lasso_lambda=2.0,
            )

            self.assertEqual(plain.dimension, 4)
            self.assertEqual(plain.X_train.shape, (8, 4))
            self.assertEqual(plain.X_test.shape, (2, 4))
            self.assertTrue(np.allclose(plain.X_train[:, -1], 1.0))
            self.assertTrue(np.allclose(np.mean(plain.X_train[:, :-1], axis=0), 0.0))
            self.assertTrue(np.allclose(plain.X_train[:, 2], 0.0))
            np.testing.assert_array_equal(np.bincount(plain.y_train.astype(int)), np.asarray([4, 4]))
            np.testing.assert_array_equal(np.bincount(plain.y_test.astype(int)), np.asarray([1, 1]))
            self.assertGreater(plain.lipschitz_constant, 0.0)
            self.assertGreater(lasso.lipschitz_constant, plain.lipschitz_constant)

            weights = np.asarray([0.75, -0.5, 0.1, 0.25], dtype=float)
            shifted_bias_weights = np.asarray([0.75, -0.5, 0.1, 1.75], dtype=float)
            self.assertGreater(lasso.objective(weights), plain.objective(weights))
            self.assertAlmostEqual(
                lasso.objective(shifted_bias_weights) - plain.objective(shifted_bias_weights),
                lasso.objective(weights) - plain.objective(weights),
                places=12,
            )
            self.assertEqual(np.asarray(plain.subgradient(weights)).shape, (4,))
            self.assertEqual(np.asarray(lasso.subgradient(weights)).shape, (4,))
            self.assertAlmostEqual(
                (
                    np.asarray(lasso.subgradient(shifted_bias_weights))
                    - np.asarray(plain.subgradient(shifted_bias_weights))
                )[-1],
                0.0,
                places=12,
            )
            self.assertGreaterEqual(plain.test_accuracy(weights), 0.0)
            self.assertLessEqual(plain.test_accuracy(weights), 1.0)
            self.assertEqual(lasso.nonzero_count(weights), 4)

    def test_train_test_split_stratifies_imbalanced_labels(self) -> None:
        X = np.arange(32, dtype=float).reshape(16, 2)
        y = np.asarray([0] * 12 + [1] * 4, dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=7)

        self.assertEqual(X_train.shape, (12, 2))
        self.assertEqual(X_test.shape, (4, 2))
        np.testing.assert_array_equal(np.bincount(y_train.astype(int)), np.asarray([9, 3]))
        np.testing.assert_array_equal(np.bincount(y_test.astype(int)), np.asarray([3, 1]))

    def test_non_binary_labels_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "bad.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["x1", "y"])
                writer.writerow([1.0, 0])
                writer.writerow([2.0, 2])

            with self.assertRaisesRegex(ValueError, "Expected binary labels"):
                build_logistic_regression_objective(dataset_path)

    def test_stratified_split_requires_at_least_two_samples_per_class(self) -> None:
        X = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
        y = np.asarray([0.0, 0.0, 1.0], dtype=float)

        with self.assertRaisesRegex(ValueError, "Stratified split requires at least two samples"):
            train_test_split(X, y, test_size=0.33, seed=0)

    def test_loader_fails_when_dropping_missing_values_leaves_one_class(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "single_class_after_drop.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["x1", "target"])
                writer.writerow([1.0, 0])
                writer.writerow(["", 1])
                writer.writerow([2.0, 0])

            with self.assertRaisesRegex(ValueError, "Dataset must contain both label classes"):
                build_logistic_regression_objective(dataset_path)

    def _write_dataset(self, dataset_path: Path) -> None:
        rows = [
            (-2.0, -1.5, 5.0, 0),
            (-1.5, -0.5, 5.0, 0),
            (-1.0, -1.0, 5.0, 0),
            (-0.5, -0.25, 5.0, 0),
            (0.5, 0.25, 5.0, 1),
            (1.0, 0.75, 5.0, 1),
            (1.5, 1.0, 5.0, 1),
            (2.0, 1.5, 5.0, 1),
            (-0.75, 0.25, 5.0, 0),
            (0.75, -0.25, 5.0, 1),
        ]
        with dataset_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["x1", "x2", "x_const", "y"])
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
