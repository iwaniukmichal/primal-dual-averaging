from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiment entrypoints.")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment name to run. Repeat to run multiple. Default: all experiments.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Experiment name to skip. Repeat to skip multiple experiments.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for outputs. Each experiment writes to OUTPUT_ROOT/<experiment>.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Forwarded to every selected experiment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected experiments and forwarded dry-run counts without writing outputs.",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Forwarded to every selected experiment to skip summary.csv generation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later experiments after one experiment process fails.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discoverable experiments and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = discover_experiments()
    if args.list:
        for experiment in experiments:
            print(experiment.name)
        return

    selected = select_experiments(
        experiments,
        include=set(args.experiment),
        exclude=set(args.exclude),
    )
    if not selected:
        raise SystemExit("No experiments selected.")

    failures: list[tuple[str, int]] = []
    for index, experiment in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {experiment.name}", flush=True)
        command = build_command(
            experiment,
            output_root=args.output_root,
            limit_runs=args.limit_runs,
            dry_run=args.dry_run,
            no_analyze=args.no_analyze,
        )
        print(" ".join(str(part) for part in command), flush=True)
        completed = subprocess.run(command, cwd=EXPERIMENTS_ROOT.parent)
        if completed.returncode == 0:
            continue
        failures.append((experiment.name, completed.returncode))
        if not args.continue_on_error:
            break

    if failures:
        summary = ", ".join(f"{name}={code}" for name, code in failures)
        raise SystemExit(f"Experiment failures: {summary}")


def discover_experiments() -> list[Path]:
    experiments: list[Path] = []
    for child in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "code" / "run.py").exists():
            experiments.append(child)
    return experiments


def select_experiments(
    experiments: list[Path],
    *,
    include: set[str],
    exclude: set[str],
) -> list[Path]:
    available = {experiment.name for experiment in experiments}
    unknown = (include | exclude) - available
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {', '.join(sorted(unknown))}")
    selected = [
        experiment
        for experiment in experiments
        if (not include or experiment.name in include) and experiment.name not in exclude
    ]
    return selected


def build_command(
    experiment: Path,
    *,
    output_root: Path | None,
    limit_runs: int | None,
    dry_run: bool,
    no_analyze: bool,
) -> list[str]:
    command = [sys.executable, str(experiment / "code" / "run.py")]
    if output_root is not None:
        command.extend(["--output-dir", str(output_root / experiment.name)])
    if limit_runs is not None:
        command.extend(["--limit-runs", str(limit_runs)])
    if dry_run:
        command.append("--dry-run")
    if no_analyze:
        command.append("--no-analyze")
    return command


if __name__ == "__main__":
    main()
