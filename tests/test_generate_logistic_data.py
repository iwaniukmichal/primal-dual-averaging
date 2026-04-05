from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "data" / "generate_logistic_data.py"


class GenerateLogisticDataCliTest(unittest.TestCase):
    def test_generator_writes_expected_csv_shape_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "synthetic.csv"
            completed = self._run_generator(
                [
                    "--n-samples",
                    "8",
                    "--dimension",
                    "3",
                    "--beta",
                    "1.0",
                    "-0.5",
                    "2.0",
                    "--seed",
                    "7",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertIn("Saved synthetic logistic data", completed.stdout)
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))

            self.assertEqual(rows[0], ["x1", "x2", "x3", "y"])
            self.assertEqual(len(rows) - 1, 8)
            for row in rows[1:]:
                self.assertEqual(len(row), 4)
                self.assertIn(row[-1], {"0", "1"})

    def test_generator_is_deterministic_for_fixed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_output = Path(temp_dir) / "first.csv"
            second_output = Path(temp_dir) / "second.csv"
            args = [
                "--n-samples",
                "10",
                "--dimension",
                "2",
                "--beta",
                "0.5",
                "-1.25",
                "--intercept",
                "0.75",
                "--seed",
                "123",
                "--output",
            ]

            self._run_generator([*args, str(first_output)])
            self._run_generator([*args, str(second_output)])

            self.assertEqual(
                first_output.read_text(encoding="utf-8"),
                second_output.read_text(encoding="utf-8"),
            )

    def test_invalid_beta_length_fails_clearly(self) -> None:
        completed = self._run_generator(
            [
                "--n-samples",
                "5",
                "--dimension",
                "3",
                "--beta",
                "1.0",
                "2.0",
            ],
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--beta length must match --dimension", completed.stderr)

    def test_invalid_flip_probability_fails_clearly(self) -> None:
        completed = self._run_generator(
            [
                "--n-samples",
                "5",
                "--dimension",
                "2",
                "--beta",
                "1.0",
                "2.0",
                "--flip-prob",
                "1.0",
            ],
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--flip-prob must satisfy 0 <= --flip-prob < 1", completed.stderr)

    def test_non_positive_sample_count_fails_clearly(self) -> None:
        completed = self._run_generator(
            [
                "--n-samples",
                "0",
                "--dimension",
                "2",
                "--beta",
                "1.0",
                "2.0",
            ],
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--n-samples must be a positive integer", completed.stderr)

    def test_non_positive_dimension_fails_clearly(self) -> None:
        completed = self._run_generator(
            [
                "--n-samples",
                "5",
                "--dimension",
                "0",
                "--beta",
                "1.0",
            ],
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--dimension must be a positive integer", completed.stderr)

    def _run_generator(
        self,
        extra_args: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *extra_args],
            cwd=PROJECT_ROOT,
            check=check,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
