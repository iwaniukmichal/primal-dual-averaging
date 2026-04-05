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
BENCH_SCRIPT_PATH = PROJECT_ROOT / "experiments" / "sda-bench.py"
LOGREG_SCRIPT_PATH = PROJECT_ROOT / "experiments" / "sda-logreg.py"
PLOT_SCRIPT_PATH = PROJECT_ROOT / "outputs" / "generate_plots.py"


class GeneratePlotsIntegrationTest(unittest.TestCase):
    def test_generates_aggregated_plots_for_bench_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bench"
            output_dir.mkdir()
            results_path = self._run_bench(output_dir, gamma_mult="1")
            self._run_bench(output_dir, gamma_mult="2")

            self._run_plot_script(results_path)

            manifest_path = output_dir / "plots" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema"], "bench")
            group = manifest["groups"][0]
            plot_ids = {plot["plot_id"] for plot in group["generated_plots"]}
            self.assertIn("objective_iterations", plot_ids)
            self.assertIn("distance_iterations", plot_ids)
            self.assertIn("iterates_iterations", plot_ids)
            self.assertIn("subgradient_subgradient_norm_iterations", plot_ids)
            self.assertIn("runtime_summary", plot_ids)
            sda_plot = next(plot for plot in group["generated_plots"] if plot["plot_id"] == "objective_iterations")
            self.assertGreaterEqual(len(sda_plot["series_labels"]), 2)
            self.assertTrue(all("<no-run-id>" not in label for label in sda_plot["series_labels"]))
            self.assertTrue(
                (output_dir / "plots" / "png" / "objective_iterations.png").exists()
            )
            self.assertTrue(
                (output_dir / "plots" / "html" / "objective_iterations.html").exists()
            )

    def test_generates_aggregated_plots_for_logreg_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "toy.csv"
            output_dir = Path(temp_dir) / "logreg"
            output_dir.mkdir()
            self._write_dataset(dataset_path)
            results_path = self._run_logreg(dataset_path, output_dir, gamma_mult="1")
            self._run_logreg(dataset_path, output_dir, gamma_mult="2")

            self._run_plot_script(results_path)

            manifest_path = output_dir / "plots" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema"], "logreg")
            group = manifest["groups"][0]
            plot_ids = {plot["plot_id"] for plot in group["generated_plots"]}
            self.assertIn("train_loss_iterations", plot_ids)
            self.assertIn("test_accuracy_iterations", plot_ids)
            self.assertIn("parameters_iterations", plot_ids)
            self.assertIn("runtime_summary", plot_ids)
            train_loss_plot = next(
                plot for plot in group["generated_plots"] if plot["plot_id"] == "train_loss_iterations"
            )
            self.assertGreaterEqual(len(train_loss_plot["series_labels"]), 2)
            self.assertTrue(
                (output_dir / "plots" / "png" / "train_loss_iterations.png").exists()
            )
            self.assertTrue(
                (output_dir / "plots" / "html" / "runtime_summary.html").exists()
            )

    def test_handles_older_bench_results_without_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bench"
            output_dir.mkdir()
            results_path = self._run_bench(output_dir, gamma_mult="1")
            runs = json.loads(results_path.read_text(encoding="utf-8"))
            for run in runs:
                run.pop("total_runtime_seconds", None)
                run.pop("avg_iteration_time_seconds", None)
            results_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

            self._run_plot_script(results_path)

            manifest_path = output_dir / "plots" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            group = manifest["groups"][0]
            skipped_ids = {entry["plot_id"] for entry in group["skipped_plots"]}
            generated_ids = {plot["plot_id"] for plot in group["generated_plots"]}

            self.assertIn("objective_time", skipped_ids)
            self.assertIn("runtime_summary", skipped_ids)
            self.assertIn("objective_iterations", generated_ids)
            self.assertTrue(
                (output_dir / "plots" / "png" / "objective_iterations.png").exists()
            )

    def test_partitions_mixed_objectives_but_not_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_one = Path(temp_dir) / "bench_one"
            source_two = Path(temp_dir) / "bench_two"
            mixed_dir = Path(temp_dir) / "mixed"
            source_one.mkdir()
            source_two.mkdir()
            mixed_dir.mkdir()

            first_results = self._run_bench(source_one, objective="weighted_l1_shift_2d", gamma_mult="1")
            second_results = self._run_bench(source_two, objective="abs_a2", gamma_mult="1")

            combined = json.loads(first_results.read_text(encoding="utf-8")) + json.loads(
                second_results.read_text(encoding="utf-8")
            )
            mixed_results = mixed_dir / "results.json"
            mixed_results.write_text(json.dumps(combined, indent=2), encoding="utf-8")

            self._run_plot_script(mixed_results)

            manifest = json.loads((mixed_dir / "plots" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["groups"]), 2)
            generated_plot_ids = {
                plot["plot_id"]
                for group in manifest["groups"]
                for plot in group["generated_plots"]
            }
            self.assertIn("weighted_l1_shift_2d__objective_iterations", generated_plot_ids)
            self.assertIn("abs_a2__objective_iterations", generated_plot_ids)
            self.assertTrue(
                (mixed_dir / "plots" / "png" / "weighted_l1_shift_2d__objective_iterations.png").exists()
            )
            self.assertTrue(
                (mixed_dir / "plots" / "png" / "abs_a2__objective_iterations.png").exists()
            )

    def _run_bench(
        self,
        output_dir: Path,
        *,
        objective: str = "weighted_l1_shift_2d",
        gamma_mult: str = "1",
    ) -> Path:
        completed = subprocess.run(
            [
                sys.executable,
                str(BENCH_SCRIPT_PATH),
                "--objective",
                objective,
                "--D",
                "2",
                "--gamma-mult",
                gamma_mult,
                "--alpha",
                "0.5",
                "-i",
                "5",
                "--output-dir",
                str(output_dir),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        self.assertIn("Results JSON path:", completed.stdout)
        return output_dir / "results.json"

    def _run_logreg(self, dataset_path: Path, output_dir: Path, *, gamma_mult: str = "1") -> Path:
        completed = subprocess.run(
            [
                sys.executable,
                str(LOGREG_SCRIPT_PATH),
                str(dataset_path),
                "--D",
                "2",
                "--gamma-mult",
                gamma_mult,
                "--alpha",
                "0.5",
                "-i",
                "5",
                "--output-dir",
                str(output_dir),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        self.assertIn("Results JSON path:", completed.stdout)
        return output_dir / "results.json"

    def _run_plot_script(self, results_path: Path) -> None:
        completed = subprocess.run(
            [sys.executable, str(PLOT_SCRIPT_PATH), str(results_path)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        self.assertIn("Generated plots for", completed.stdout)

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
