from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "sda-logreg.py"


class LogisticRegressionExperimentIntegrationTest(unittest.TestCase):
    def test_cli_runs_for_plain_and_lasso_modes(self) -> None:
        plain_results = self._run_script([])
        lasso_results = self._run_script(["--lasso", "--lambda", "2.0"])

        self.assertEqual([run["method"] for run in plain_results], ["sda", "subgradient", "sklearn"])
        self.assertEqual([run["method"] for run in lasso_results], ["sda", "subgradient", "sklearn"])

        plain_by_method = {run["method"]: run for run in plain_results}
        lasso_by_method = {run["method"]: run for run in lasso_results}

        plain_sda = plain_by_method["sda"]
        plain_subgradient = plain_by_method["subgradient"]
        plain_sklearn = plain_by_method["sklearn"]
        lasso_sda = lasso_by_method["sda"]
        lasso_subgradient = lasso_by_method["subgradient"]
        lasso_sklearn = lasso_by_method["sklearn"]

        self.assertEqual(plain_sda["objective_family"], "logistic_regression")
        self.assertEqual(plain_subgradient["objective_family"], "logistic_regression")
        self.assertEqual(lasso_sda["objective_family"], "lasso_logistic_regression")
        self.assertEqual(lasso_subgradient["objective_family"], "lasso_logistic_regression")
        self.assertEqual(plain_sda["solver_name"], "sda")
        self.assertEqual(plain_subgradient["solver_name"], "subgradient")
        self.assertEqual(plain_sklearn["solver_name"], "saga")
        self.assertEqual(lasso_sklearn["solver_name"], "saga")
        self.assertIn("gamma_multiplier", plain_sda)
        self.assertIn("alpha", plain_subgradient)
        self.assertIn("alpha_k", plain_subgradient)
        self.assertIn("g_norm", plain_subgradient)
        self.assertIn("final_parameter_vector", plain_sklearn)
        self.assertIsNone(plain_sklearn["sklearn_penalty"])
        self.assertFalse(plain_sklearn["restrict_to_fd"])
        self.assertTrue(lasso_sda["objective_params"]["lasso"])
        self.assertTrue(lasso_subgradient["objective_params"]["lasso"])
        self.assertEqual(lasso_sklearn["sklearn_penalty"], "l1")

    def test_deduplicates_results_json_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "toy.csv"
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            self._write_dataset(dataset_path)

            first = self._run_into_output_dir(dataset_path, output_dir, [])
            second = self._run_into_output_dir(dataset_path, output_dir, [])

            self.assertEqual([run["method"] for run in first], ["sda", "subgradient", "sklearn"])
            self.assertEqual(len(second), 3)
            self.assertEqual({run["run_id"] for run in second}, {first[0]["run_id"]})

    def test_default_output_dir_is_dataset_and_mode_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "toy.csv"
            self._write_dataset(dataset_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    str(dataset_path),
                    "--D",
                    "2",
                    "--alpha",
                    "0.5",
                    "-i",
                    "5",
                ],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

            results_path = (
                Path(temp_dir)
                / "outputs"
                / "sda-logreg"
                / "toy"
                / "logreg"
                / "results.json"
            )
            self.assertIn(
                "Results JSON path: outputs/sda-logreg/toy/logreg/results.json",
                completed.stdout,
            )
            self.assertTrue(results_path.exists())
            self.assertEqual(len(json.loads(results_path.read_text(encoding="utf-8"))), 3)

    def test_malformed_existing_results_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "toy.csv"
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            self._write_dataset(dataset_path)
            (output_dir / "results.json").write_text('{"bad": true}', encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    str(dataset_path),
                    "--D",
                    "2",
                    "--gamma-mult",
                    "1",
                    "--alpha",
                    "0.5",
                    "-i",
                    "5",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Existing results file must contain a JSON array", completed.stderr)

    def _run_script(self, extra_args: list[str]) -> list[dict[str, object]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "toy.csv"
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            self._write_dataset(dataset_path)
            return self._run_into_output_dir(dataset_path, output_dir, extra_args)

    def _run_into_output_dir(
        self,
        dataset_path: Path,
        output_dir: Path,
        extra_args: list[str],
    ) -> list[dict[str, object]]:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                str(dataset_path),
                "--D",
                "2",
                "--gamma-mult",
                "1",
                "--alpha",
                "0.5",
                "-i",
                "5",
                "--output-dir",
                str(output_dir),
                *extra_args,
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        self.assertIn("Results JSON path:", completed.stdout)
        results_path = output_dir / "results.json"
        self.assertTrue(results_path.exists())
        return json.loads(results_path.read_text(encoding="utf-8"))

    def _write_dataset(self, dataset_path: Path) -> None:
        rows = [
            (-2.0, -1.5, 0),
            (-1.5, -0.5, 0),
            (-1.0, -1.0, 0),
            (-0.5, -0.25, 0),
            (0.5, 0.25, 1),
            (1.0, 0.75, 1),
            (1.5, 1.0, 1),
            (2.0, 1.5, 1),
            (-0.75, 0.25, 0),
            (0.75, -0.25, 1),
        ]
        with dataset_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["x1", "x2", "y"])
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
