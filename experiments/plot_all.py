from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiment plot entrypoints.")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment name to plot. Repeat to plot multiple. Default: all experiments with code/plot.py.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Experiment name to skip. Repeat to skip multiple experiments.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for input CSVs. Each plot.py receives "
            "--results-dir RESULTS_ROOT/<experiment>."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for plot outputs. Each plot.py receives "
            "--out-dir OUT_ROOT/<experiment>."
        ),
    )
    parser.add_argument(
        "--objective",
        action="append",
        default=[],
        help="Forwarded objective filter. Repeat to include multiple objectives.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Forwarded primary trajectory metric.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=[],
        help="Forwarded plot group selector, for example: --plots gaps norms.",
    )
    parser.add_argument(
        "--max-traces-per-objective",
        type=int,
        default=None,
        help="Forwarded maximum trace count per objective.",
    )
    parser.add_argument(
        "--max-points-per-run",
        type=int,
        default=None,
        help="Forwarded maximum points per trajectory run.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Forwarded to open generated HTML files in the default browser.",
    )
    parser.add_argument(
        "--show-traces",
        action="store_true",
        help="Forwarded to make trajectory traces visible by default.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Additional raw argument forwarded to every plot.py. Repeat for each token, "
            "for example: --extra-arg=--some-flag --extra-arg=value."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running plot scripts.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later plot scripts after one process fails.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discoverable plot experiments and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = discover_plot_experiments()
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
        raise SystemExit("No plot scripts selected.")

    failures: list[tuple[str, int]] = []
    for index, experiment in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {experiment.name}", flush=True)
        command = build_command(
            experiment,
            results_root=args.results_root,
            out_root=args.out_root,
            objectives=args.objective,
            metric=args.metric,
            plots=args.plots,
            max_traces_per_objective=args.max_traces_per_objective,
            max_points_per_run=args.max_points_per_run,
            show=args.show,
            show_traces=args.show_traces,
            extra_args=args.extra_arg,
        )
        print(" ".join(str(part) for part in command), flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=EXPERIMENTS_ROOT.parent)
        if completed.returncode == 0:
            continue
        failures.append((experiment.name, completed.returncode))
        if not args.continue_on_error:
            break

    if failures:
        summary = ", ".join(f"{name}={code}" for name, code in failures)
        raise SystemExit(f"Plot failures: {summary}")


def discover_plot_experiments() -> list[Path]:
    experiments: list[Path] = []
    for child in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "code" / "plot.py").exists():
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
        raise SystemExit(f"Unknown plot experiment(s): {', '.join(sorted(unknown))}")
    return [
        experiment
        for experiment in experiments
        if (not include or experiment.name in include) and experiment.name not in exclude
    ]


def build_command(
    experiment: Path,
    *,
    results_root: Path | None,
    out_root: Path | None,
    objectives: list[str],
    metric: str | None,
    plots: list[str],
    max_traces_per_objective: int | None,
    max_points_per_run: int | None,
    show: bool,
    show_traces: bool,
    extra_args: list[str],
) -> list[str]:
    command = [sys.executable, str(experiment / "code" / "plot.py")]
    if results_root is not None:
        command.extend(["--results-dir", str(results_root / experiment.name)])
    if out_root is not None:
        command.extend(["--out-dir", str(out_root / experiment.name)])
    for objective in objectives:
        command.extend(["--objective", objective])
    if metric is not None:
        command.extend(["--metric", metric])
    if plots:
        command.append("--plots")
        command.extend(plots)
    if max_traces_per_objective is not None:
        command.extend(["--max-traces-per-objective", str(max_traces_per_objective)])
    if max_points_per_run is not None:
        command.extend(["--max-points-per-run", str(max_points_per_run)])
    if show:
        command.append("--show")
    if show_traces:
        command.append("--show-traces")
    command.extend(extra_args)
    return command


if __name__ == "__main__":
    main()
