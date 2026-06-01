from __future__ import annotations

from pathlib import Path
from typing import Any

from .runners import analyze_results, build_parser, default_results_dir, execute_experiment


def run_experiment_main(
    *,
    experiment: str,
    description: str,
    configs: list[dict[str, Any]],
    runner_kind: str,
    run_file: str | Path,
) -> None:
    parser = build_parser(description)
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip automatic summary.csv generation after the run.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or default_results_dir(run_file)
    execute_experiment(
        experiment=experiment,
        configs=configs,
        output_dir=output_dir,
        runner_kind=runner_kind,
        dry_run=bool(args.dry_run),
        limit_runs=args.limit_runs,
    )
    if not args.dry_run and not args.no_analyze:
        analyze_results(output_dir)


def analyze_experiment_main(*, run_file: str | Path) -> None:
    parser = build_parser("Analyze experiment CSV results.")
    args = parser.parse_args()
    output_dir = args.output_dir or default_results_dir(run_file)
    if args.dry_run:
        print(f"Would analyze {output_dir}")
        return
    analyze_results(output_dir)
