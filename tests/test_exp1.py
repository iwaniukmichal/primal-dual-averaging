from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP1_PATH = PROJECT_ROOT / "experiments" / "exp1.py"


class ExperimentIntegrationTest(unittest.TestCase):
    def test_no_plot_cli_runs_for_scalar_and_vector_objectives(self) -> None:
        scalar_results = self._run_exp1(["--objective", "weighted_l1_shift_1d", "--no-plot"])
        vector_results = self._run_exp1(["--objective", "weighted_l1_shift_2d", "--no-plot"])

        self.assertEqual(scalar_results[0]["objective_id"], "weighted_l1_shift_1d")
        self.assertEqual(vector_results[0]["objective_id"], "weighted_l1_shift_2d")
        self.assertIsInstance(vector_results[0]["x"][0], list)
        self.assertIn("x_norm", vector_results[0])
        self.assertIn("x_hat_norm", vector_results[0])
        self.assertIn("objective_minimum_value", vector_results[0])
        self.assertIn("objective_minimizer", vector_results[0])
        self.assertIn("objective_minimizer_norm", vector_results[0])
        self.assertIn("final_objective_gap_x_hat", vector_results[0])

    def test_plot_cli_runs_for_vector_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"
            mpl_config_dir = Path(temp_dir) / "mpl"
            output_dir.mkdir()
            mpl_config_dir.mkdir()

            env = os.environ.copy()
            env.update(
                {
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": str(mpl_config_dir),
                }
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(EXP1_PATH),
                    "--objective",
                    "max_affine_4d",
                    "--D",
                    "6",
                    "--gamma-mult",
                    "1",
                    "-i",
                    "5",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertIn("Saved plots:", completed.stdout)
            self.assertTrue((output_dir / "exp1_D_6.png").exists())

            runs = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertIsInstance(runs[0]["x"][0], list)
            self.assertIsInstance(runs[0]["objective_minimizer"], list)
            self.assertIn("x_norm", runs[0])
            self.assertIn("x_hat_norm", runs[0])
            self.assertIn("objective_minimum_value", runs[0])
            self.assertIn("final_objective_gap_x", runs[0])
            self.assertIn("final_objective_gap_x_hat", runs[0])

    def test_default_output_dir_is_objective_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mpl_config_dir = Path(temp_dir) / "mpl"
            mpl_config_dir.mkdir()

            env = os.environ.copy()
            env.update(
                {
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": str(mpl_config_dir),
                }
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(EXP1_PATH),
                    "--objective",
                    "weighted_l1_shift_2d",
                    "--D",
                    "2",
                    "--gamma-mult",
                    "1",
                    "-i",
                    "5",
                    "--no-plot",
                ],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            results_path = Path(temp_dir) / "outputs" / "exp1" / "weighted_l1_shift_2d" / "results.json"
            self.assertIn(
                "Saved JSON summary to outputs/exp1/weighted_l1_shift_2d/results.json",
                completed.stdout,
            )
            self.assertTrue(results_path.exists())

    def _run_exp1(self, extra_args: list[str]) -> list[dict[str, object]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"
            mpl_config_dir = Path(temp_dir) / "mpl"
            output_dir.mkdir()
            mpl_config_dir.mkdir()

            env = os.environ.copy()
            env.update(
                {
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": str(mpl_config_dir),
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EXP1_PATH),
                    "--D",
                    "2",
                    "--gamma-mult",
                    "1",
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
                env=env,
            )

            self.assertIn("Saved JSON summary", completed.stdout)
            results_path = output_dir / "results.json"
            self.assertTrue(results_path.exists())
            return json.loads(results_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
