from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "sda-bench.py"


class SDABenchIntegrationTest(unittest.TestCase):
    def test_cli_runs_and_emits_two_methods(self) -> None:
        results = self._run_script(["--objective", "weighted_l1_shift_2d"])

        self.assertEqual([run["method"] for run in results], ["sda", "subgradient"])

        sda_run = results[0]
        subgradient_run = results[1]

        self.assertEqual(sda_run["objective_id"], "weighted_l1_shift_2d")
        self.assertEqual(subgradient_run["objective_id"], "weighted_l1_shift_2d")
        self.assertIsInstance(sda_run["x"][0], list)
        self.assertIsInstance(subgradient_run["x"][0], list)
        self.assertIn("gamma_multiplier", sda_run)
        self.assertIn("gamma_star", sda_run)
        self.assertIn("normalized_gap", sda_run)
        self.assertIn("total_runtime_seconds", sda_run)
        self.assertIn("avg_iteration_time_seconds", sda_run)
        self.assertIn("run_id", sda_run)
        self.assertIn("alpha", subgradient_run)
        self.assertIn("alpha_k", subgradient_run)
        self.assertIn("g_norm", subgradient_run)
        self.assertIn("total_runtime_seconds", subgradient_run)
        self.assertIn("avg_iteration_time_seconds", subgradient_run)
        self.assertIn("run_id", subgradient_run)
        self.assertNotIn("gamma_multiplier", subgradient_run)
        self.assertNotIn("gamma_star", subgradient_run)
        self.assertNotIn("normalized_gap", subgradient_run)

    def test_deduplicates_results_json_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()

            first = self._run_into_output_dir(output_dir, ["--objective", "weighted_l1_shift_2d"])
            second = self._run_into_output_dir(output_dir, ["--objective", "weighted_l1_shift_2d"])

            self.assertEqual([run["method"] for run in first], ["sda", "subgradient"])
            self.assertEqual(len(second), 2)
            self.assertEqual({run["run_id"] for run in second}, {first[0]["run_id"]})

    def test_malformed_existing_results_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            (output_dir / "results.json").write_text('{"bad": true}', encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--objective",
                    "weighted_l1_shift_2d",
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
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            return self._run_into_output_dir(output_dir, extra_args)

    def _run_into_output_dir(
        self,
        output_dir: Path,
        extra_args: list[str],
    ) -> list[dict[str, object]]:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
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


if __name__ == "__main__":
    unittest.main()
