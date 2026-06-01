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

EXPERIMENTS = [
    "nonsmooth_objective_grid",
    "prox_geometry_comparison",
    "simplex_entropy_geometry",
    "logistic_regression_prediction",
    "lasso_logistic_sparsity",
    "runtime_scaling",
    "stopping_and_fd_ablation",
]


class ReportExperimentSuiteTest(unittest.TestCase):
    def test_experiment_folders_have_required_files(self) -> None:
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                root = PROJECT_ROOT / "experiments" / experiment
                self.assertTrue((root / "description.md").exists())
                self.assertTrue((root / "code" / "run.py").exists())
                self.assertFalse((root / "code" / "analyze.py").exists())
                self.assertTrue((root / "results" / ".gitkeep").exists())

    def test_dry_runs_report_configured_counts(self) -> None:
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "experiments" / experiment / "code" / "run.py"),
                        "--dry-run",
                        "--limit-runs",
                        "2",
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                self.assertIn(experiment, completed.stdout)
                self.assertIn("configured runs", completed.stdout)

    def test_limited_runs_write_csv_metadata_and_summary(self) -> None:
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_dir = Path(temp_dir) / "results"
                    run_script = PROJECT_ROOT / "experiments" / experiment / "code" / "run.py"

                    subprocess.run(
                        [
                            sys.executable,
                            str(run_script),
                            "--limit-runs",
                            "3",
                            "--output-dir",
                            str(output_dir),
                        ],
                        cwd=PROJECT_ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=os.environ.copy(),
                    )
                    self.assertTrue((output_dir / "summary.csv").exists())

                    runs_path = output_dir / "runs.csv"
                    iterations_path = output_dir / "iterations.csv"
                    metadata_path = output_dir / "metadata.json"
                    summary_path = output_dir / "summary.csv"
                    self.assertTrue(runs_path.exists())
                    self.assertTrue(iterations_path.exists())
                    self.assertTrue(metadata_path.exists())
                    self.assertTrue(summary_path.exists())

                    with runs_path.open("r", encoding="utf-8", newline="") as handle:
                        runs = list(csv.DictReader(handle))
                    self.assertGreater(len(runs), 0)
                    self.assertIn("run_id", runs[0])
                    self.assertIn("runtime_seconds", runs[0])

                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    self.assertEqual(metadata["experiment"], experiment)
                    self.assertEqual(metadata["selected_run_count"], 3)


if __name__ == "__main__":
    unittest.main()
