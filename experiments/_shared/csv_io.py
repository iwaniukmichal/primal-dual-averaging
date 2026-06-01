from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RUN_COLUMNS = [
    "experiment",
    "run_id",
    "method",
    "solver_family",
    "objective_id",
    "dataset",
    "objective_family",
    "dimension",
    "prox_fun",
    "prox_weights_profile",
    "dual_averaging",
    "D",
    "gamma",
    "gamma_mult",
    "alpha",
    "lambda",
    "eps",
    "max_iter",
    "restrict_to_fd",
    "seed",
    "iterations",
    "converged",
    "runtime_seconds",
    "avg_iteration_time_seconds",
    "objective_gap",
    "train_loss",
    "test_loss",
    "test_accuracy",
    "nonzero_count",
    "final_norm",
    "final_gap",
    "final_x",
    "final_x_hat",
    "final_parameter_vector",
]

ITERATION_COLUMNS = [
    "run_id",
    "iteration",
    "elapsed_estimate_seconds",
    "objective_value_x",
    "objective_value_x_hat",
    "objective_gap_x",
    "objective_gap_x_hat",
    "normalized_gap",
    "x_norm",
    "x_hat_norm",
    "g_norm",
    "nonzero_count",
]


def json_cell(value: Any) -> str:
    """Serialize complex CSV cells deterministically."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows using a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: json_cell(row.get(column)) for column in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

