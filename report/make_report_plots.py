from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pda_report_mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pda_report_cache")
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_OUT_DIR = PROJECT_ROOT / "report" / "figures"

PLOT_CHOICES = {"all", "nonsmooth", "lasso", "lasso-ssa", "prox", "simplex", "runtime-gap"}
DEFAULT_FORMATS = ("pdf", "png")

METHOD_LABELS = {
    "sda_simple_euclidean": "SDA simple",
    "sda_weighted_euclidean_dual_average": "SDA weighted",
    "sda_weighted_euclidean": "SDA weighted",
    "ssa_euclidean": "SSA",
    "projected_subgradient": "Projected subgradient",
    "sklearn_saga": "sklearn SAGA",
    "sda_simple_weighted_euclidean": "SDA simple, weighted prox",
    "sda_weighted_weighted_euclidean": "SDA weighted, weighted prox",
    "sda_simple_entropy": "SDA simple entropy",
    "sda_weighted_entropy": "SDA weighted entropy",
}

METHOD_COLORS = {
    "sda_simple_euclidean": "#1f77b4",
    "sda_weighted_euclidean_dual_average": "#d62728",
    "sda_weighted_euclidean": "#d62728",
    "ssa_euclidean": "#1f77b4",
    "projected_subgradient": "#2ca02c",
    "sklearn_saga": "#111111",
    "sda_simple_weighted_euclidean": "#9467bd",
    "sda_weighted_weighted_euclidean": "#ff7f0e",
    "sda_simple_entropy": "#1f77b4",
    "sda_weighted_entropy": "#d62728",
}

D_STYLES = {
    0.5: "solid",
    1.0: "solid",
    2.0: "dashed",
    3.0: "dotted",
    8.0: "dashdot",
    32.0: (0, (5, 1)),
    128.0: (0, (1, 1)),
}

GAMMA_COLORS = {
    0.1: "#1f77b4",
    0.25: "#ff7f0e",
    0.5: "#2ca02c",
    1.0: "#d62728",
    2.0: "#9467bd",
    4.0: "#17becf",
}

MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#e377c2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create static paper-ready figures for the experiment report."
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
        help=f"Directory containing experiment folders. Default: {DEFAULT_EXPERIMENTS_ROOT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory where report figures are written. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "png"],
        default=list(DEFAULT_FORMATS),
        help="Output formats. Default: pdf png.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=sorted(PLOT_CHOICES),
        default=["all"],
        help="Plot groups to generate. Default: all.",
    )
    parser.add_argument(
        "--max-points-per-run",
        type=int,
        default=300,
        help="Maximum trajectory points per run. Use 0 to keep all points. Default: 300.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    groups = expand_plot_choices(args.plots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if should_run("nonsmooth", groups):
        written.extend(
            write_nonsmooth_figures(
                args.experiments_root / "nonsmooth_objective_grid",
                args.out_dir / "nonsmooth_objective_grid",
                args.formats,
                args.max_points_per_run,
                runtime_only="runtime-gap" in groups and "nonsmooth" not in groups,
            )
        )
    if should_run("lasso", groups):
        written.extend(
            write_lasso_figures(
                args.experiments_root / "lasso_logistic_sparsity",
                args.out_dir / "lasso_logistic_sparsity",
                args.formats,
                args.max_points_per_run,
                runtime_only="runtime-gap" in groups and "lasso" not in groups,
            )
        )
    if should_run("lasso-ssa", groups):
        written.extend(
            write_lasso_ssa_figures(
                args.experiments_root / "lasso_logistic_sparsity_ssa_comparison",
                args.out_dir / "lasso_logistic_sparsity_ssa_comparison",
                args.formats,
                args.max_points_per_run,
                runtime_only="runtime-gap" in groups and "lasso-ssa" not in groups,
            )
        )
    if should_run("prox", groups):
        written.extend(
            write_prox_figures(
                args.experiments_root / "prox_geometry_comparison",
                args.out_dir / "prox_geometry_comparison",
                args.formats,
                args.max_points_per_run,
                runtime_only="runtime-gap" in groups and "prox" not in groups,
            )
        )
    if should_run("simplex", groups):
        written.extend(
            write_simplex_figures(
                args.experiments_root / "simplex_entropy_geometry",
                args.out_dir / "simplex_entropy_geometry",
                args.formats,
                args.max_points_per_run,
                runtime_only="runtime-gap" in groups and "simplex" not in groups,
            )
        )

    print(f"Wrote {len(written)} figure file(s) under {args.out_dir}")
    for path in written:
        print(path)


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.45,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def expand_plot_choices(values: Iterable[str]) -> set[str]:
    groups = set(values)
    if "all" in groups:
        return {"nonsmooth", "lasso", "prox", "simplex"}
    return groups


def should_run(group: str, groups: set[str]) -> bool:
    return group in groups or "runtime-gap" in groups


def write_nonsmooth_figures(
    exp_dir: Path,
    out_dir: Path,
    formats: list[str],
    max_points: int,
    *,
    runtime_only: bool,
) -> list[Path]:
    runs, iterations = load_runs_and_iterations(exp_dir)
    runs = add_common_metadata(runs)
    data = merge_iterations(iterations, runs, max_points)
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    written.extend(
        save_runtime_gap_grid(
            runs,
            out_dir,
            "nonsmooth_runtime_vs_final_objective_gap",
            formats,
            group_col="objective_id",
            y_col="objective_gap",
            title="Runtime vs final objective gap on benchmark objectives",
        )
    )
    if runtime_only:
        return written

    subset = data[
        (data["method"] == "sda_simple_euclidean")
        & near(data["gamma_mult"], 1.0)
        & (data["restrict_to_fd"] == False)
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "nonsmooth_default_unrestricted",
            formats,
            title="SDA simple, unrestricted, gamma_mult=1",
            group_col="objective_id",
            y_col="objective_gap_x_hat",
            y_label="Objective gap at averaged iterate",
            label_func=lambda row: f"D={fmt(row['D'])}",
            color_func=lambda row: color_by_value(row["D"]),
            style_func=lambda row: D_STYLES.get(float(row["D"]), "solid"),
        )
    )

    subset = data[
        (data["method"] == "sda_simple_euclidean")
        & near(data["gamma_mult"], 1.0)
        & (data["restrict_to_fd"] == True)
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "nonsmooth_default_restricted",
            formats,
            title="SDA simple, restricted to F_D, gamma_mult=1",
            group_col="objective_id",
            y_col="objective_gap_x_hat",
            y_label="Objective gap at averaged iterate",
            label_func=lambda row: f"D={fmt(row['D'])}",
            color_func=lambda row: color_by_value(row["D"]),
            style_func=lambda row: D_STYLES.get(float(row["D"]), "solid"),
        )
    )

    subset = data[
        (data["method"] == "sda_simple_euclidean")
        & near(data["gamma_mult"], 0.1)
        & near(data["D"], 8.0)
    ]
    written.extend(
        save_multi_metric_grid_figure(
            subset,
            out_dir,
            "nonsmooth_restriction_norm_gap",
            formats,
            title="SDA simple, D=8, gamma_mult=0.1: restriction effect",
            group_col="objective_id",
            metrics=["objective_gap_x", "objective_gap_x_hat"],
            y_label="Objective gap",
            label_func=lambda row, metric: (
                f"{'restricted' if row['restrict_to_fd'] else 'unrestricted'}, {metric}"
            ),
            color_func=lambda row, metric: "#1f77b4"
            if not bool(row["restrict_to_fd"])
            else "#d62728",
            style_func=lambda row, metric: "solid"
            if metric == "objective_gap_x_hat"
            else "dashed",
        )
    )

    subset = data[
        (data["method"] == "sda_simple_euclidean")
        & near(data["D"], 8.0)
        & (data["restrict_to_fd"] == False)
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "nonsmooth_gamma_sweep",
            formats,
            title="SDA simple gamma sweep, unrestricted, D=8",
            group_col="objective_id",
            y_col="objective_gap_x_hat",
            y_label="Objective gap at averaged iterate",
            label_func=lambda row: f"gamma_mult={fmt(row['gamma_mult'])}",
            color_func=lambda row: gamma_color(row["gamma_mult"]),
            style_func=lambda row: "solid",
        )
    )

    subset = data[
        (data["method"] == "sda_simple_euclidean")
        & data["gamma_mult"].isin([0.5, 2.0])
        & (data["restrict_to_fd"] == False)
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "nonsmooth_gamma_d_grid",
            formats,
            title="SDA simple gamma and D interaction, unrestricted",
            group_col="objective_id",
            y_col="objective_gap_x_hat",
            y_label="Objective gap at averaged iterate",
            label_func=lambda row: f"gamma_mult={fmt(row['gamma_mult'])}, D={fmt(row['D'])}",
            color_func=lambda row: gamma_color(row["gamma_mult"]),
            style_func=lambda row: D_STYLES.get(float(row["D"]), "solid"),
        )
    )

    subset = data[
        (data["restrict_to_fd"] == False)
        & data["D"].isin([2.0, 8.0])
        & (
            (
                data["method"].isin(
                    [
                        "sda_simple_euclidean",
                        "sda_weighted_euclidean_dual_average",
                    ]
                )
                & near(data["gamma_mult"], 1.0)
            )
            | (
                (data["method"] == "projected_subgradient")
                & near(data["alpha"], 1.0)
            )
        )
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "nonsmooth_method_comparison",
            formats,
            title="SDA simple, weighted SDA, and subgradient comparison",
            group_col="objective_id",
            y_col="objective_gap_x_hat",
            y_label="Objective gap at averaged iterate",
            label_func=lambda row: f"{method_label(row['method'])}, D={fmt(row['D'])}",
            color_func=lambda row: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row: D_STYLES.get(float(row["D"]), "solid"),
        )
    )

    return written


def write_lasso_figures(
    exp_dir: Path,
    out_dir: Path,
    formats: list[str],
    max_points: int,
    *,
    runtime_only: bool,
) -> list[Path]:
    runs, iterations = load_runs_and_iterations(exp_dir)
    runs = add_common_metadata(add_lasso_sklearn_references(runs))
    custom_runs = runs[runs["method"] != "sklearn_saga"].copy()
    data = merge_iterations(iterations, custom_runs, max_points)
    data["objective_gap_to_sklearn_x_hat"] = (
        data["objective_value_x_hat"] - data["sklearn_train_loss"]
    )
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    written.extend(
        save_runtime_gap_grid(
            runs,
            out_dir,
            "lasso_runtime_vs_final_objective_gap_to_sklearn",
            formats,
            group_col="dataset_label",
            y_col="objective_gap_to_sklearn",
            title="Runtime vs final train objective gap to sklearn",
        )
    )
    if runtime_only:
        return written

    subset = data[
        near(data["lambda"], 1.0)
        & near(data["seed"], 0.0)
        & (
            (
                data["method"].isin(["sda_simple_euclidean", "sda_weighted_euclidean"])
                & near(data["gamma_mult"], 1.0)
            )
            | (
                (data["method"] == "projected_subgradient")
                & near(data["alpha"], 1.0)
            )
        )
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "lasso_d_comparison",
            formats,
            title="Lasso logistic objective gap to sklearn, lambda=1",
            group_col="dataset_label",
            y_col="objective_gap_to_sklearn_x_hat",
            y_label="Train objective gap to sklearn",
            label_func=lambda row: f"{method_label(row['method'])}, D={fmt(row['D'])}",
            color_func=lambda row: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row: D_STYLES.get(float(row["D"]), "solid"),
            zero_line=True,
        )
    )

    subset = data[
        near(data["lambda"], 1.0)
        & near(data["seed"], 0.0)
        & near(data["D"], 32.0)
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "lasso_parameter_sweep",
            formats,
            title="Lasso logistic parameter sweep, lambda=1, D=32",
            group_col="dataset_label",
            y_col="objective_gap_to_sklearn_x_hat",
            y_label="Train objective gap to sklearn",
            label_func=lambda row: tuning_label(row),
            color_func=lambda row: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row: tuning_style(row),
            zero_line=True,
        )
    )
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "lasso_parameter_sweep_train_objective",
            formats,
            title="Lasso logistic train objective, lambda=1, D=32",
            group_col="dataset_label",
            y_col="objective_value_x_hat",
            y_label="Train objective",
            label_func=lambda row: tuning_label(row),
            color_func=lambda row: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row: tuning_style(row),
            reference_func=single_reference("sklearn_train_loss"),
            reference_label="sklearn train objective",
            force_linear=True,
            log_x=True,
        )
    )

    subset = data[
        near(data["lambda"], 1.0)
        & near(data["seed"], 0.0)
        & (
            (
                data["method"].isin(["sda_simple_euclidean", "sda_weighted_euclidean"])
                & near(data["gamma_mult"], 1.0)
            )
            | (
                (data["method"] == "projected_subgradient")
                & near(data["alpha"], 1.0)
            )
        )
    ]
    written.extend(
        save_multi_metric_grid_figure(
            subset,
            out_dir,
            "lasso_norms",
            formats,
            title="Lasso logistic iterate norms, lambda=1",
            group_col="dataset_label",
            metrics=["x_hat_norm"],
            y_label="Euclidean norm",
            label_func=lambda row, metric: f"{method_label(row['method'])}, D={fmt(row['D'])}",
            color_func=lambda row, metric: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row, metric: D_STYLES.get(float(row["D"]), "solid"),
            reference_func=single_reference("sklearn_norm"),
            reference_label=r"$\|x_{\mathrm{sklearn}}\|$",
            force_linear=True,
        )
    )

    return written


def write_lasso_ssa_figures(
    exp_dir: Path,
    out_dir: Path,
    formats: list[str],
    max_points: int,
    *,
    runtime_only: bool,
) -> list[Path]:
    runs, iterations = load_runs_and_iterations(exp_dir)
    runs = runs[runs["dataset"].map(is_synthetic_dataset)].copy()
    runs = add_common_metadata(add_lasso_sklearn_references(runs))
    comparison_methods = ["sda_weighted_euclidean", "ssa_euclidean"]
    custom_runs = runs[runs["method"].isin(comparison_methods)].copy()
    data = merge_iterations(iterations, custom_runs, max_points)
    data["objective_gap_to_sklearn_x_hat"] = (
        data["objective_value_x_hat"] - data["sklearn_train_loss"]
    )
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    written.extend(
        save_runtime_gap_grid(
            runs[runs["method"].isin(comparison_methods + ["sklearn_saga"])].copy(),
            out_dir,
            "lasso_ssa_runtime_vs_final_objective_gap_to_sklearn",
            formats,
            group_col="dataset_label",
            y_col="objective_gap_to_sklearn",
            title="Weighted SDA vs SSA runtime and final train objective gap",
        )
    )
    if runtime_only:
        return written

    available_D = sorted(float(value) for value in data["D"].dropna().unique())
    sweep_D = 32.0 if any(abs(value - 32.0) <= 1e-9 for value in available_D) else available_D[0]

    subset = data[
        near(data["lambda"], 1.0)
        & near(data["seed"], 0.0)
        & near(data["D"], sweep_D)
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "lasso_ssa_parameter_sweep",
            formats,
            title=f"Weighted SDA vs SSA parameter sweep, lambda=1, D={fmt(sweep_D)}",
            group_col="dataset_label",
            y_col="objective_gap_to_sklearn_x_hat",
            y_label="Train objective gap to sklearn",
            label_func=lambda row: tuning_label(row),
            color_func=lambda row: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row: tuning_style(row),
            zero_line=True,
        )
    )
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "lasso_ssa_parameter_sweep_train_objective",
            formats,
            title=f"Weighted SDA vs SSA train objective, lambda=1, D={fmt(sweep_D)}",
            group_col="dataset_label",
            y_col="objective_value_x_hat",
            y_label="Train objective",
            label_func=lambda row: tuning_label(row),
            color_func=lambda row: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row: tuning_style(row),
            reference_func=single_reference("sklearn_train_loss"),
            reference_label="sklearn train objective",
            force_log=True,
        )
    )
    return written


def is_synthetic_dataset(value: object) -> bool:
    return "synthetic" in dataset_label(value)


def write_prox_figures(
    exp_dir: Path,
    out_dir: Path,
    formats: list[str],
    max_points: int,
    *,
    runtime_only: bool,
) -> list[Path]:
    runs, iterations = load_runs_and_iterations(exp_dir)
    runs = add_common_metadata(runs)
    data = merge_iterations(iterations, runs, max_points)
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    written.extend(
        save_runtime_gap_grid(
            runs,
            out_dir,
            "prox_runtime_vs_final_objective_gap",
            formats,
            group_col="objective_id",
            y_col="objective_gap",
            title="Runtime vs final objective gap by prox geometry",
        )
    )
    if runtime_only:
        return written

    subset = data[
        near(data["D"], 32.0)
        & near(data["gamma_mult"], 1.0)
        & (data["restrict_to_fd"] == False)
    ]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "prox_geometry",
            formats,
            title="Prox geometry comparison, D=32, gamma_mult=1, unrestricted",
            group_col="objective_id",
            y_col="objective_gap_x_hat",
            y_label="Objective gap at averaged iterate",
            label_func=lambda row: prox_label(row),
            color_func=lambda row: prox_color(row),
            style_func=lambda row: prox_style(row),
        )
    )
    return written


def write_simplex_figures(
    exp_dir: Path,
    out_dir: Path,
    formats: list[str],
    max_points: int,
    *,
    runtime_only: bool,
) -> list[Path]:
    runs, iterations = load_runs_and_iterations(exp_dir)
    runs = add_common_metadata(runs)
    data = merge_iterations(iterations, runs, max_points)
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    written.extend(
        save_runtime_gap_grid(
            runs,
            out_dir,
            "simplex_runtime_vs_final_objective_gap",
            formats,
            group_col="objective_id",
            y_col="objective_gap",
            title="Runtime vs final objective gap with entropy prox",
        )
    )
    if runtime_only:
        return written

    subset = data[data["D"].isin([2, 3]) & data["gamma_mult"].isin([0.1, 0.5, 1.0])]
    written.extend(
        save_line_grid_figure(
            subset,
            out_dir,
            "simplex_entropy",
            formats,
            title="Entropy SDA trajectories, D in {2, 3}",
            group_col="objective_id",
            y_col="objective_gap_x_hat",
            y_label="Objective gap at averaged iterate",
            label_func=lambda row: (
                f"{method_label(row['method'])}, "
                f"gamma_mult={fmt(row['gamma_mult'])}, D={fmt(row['D'])}"
            ),
            color_func=lambda row: METHOD_COLORS.get(str(row["method"]), "#555555"),
            style_func=lambda row: D_STYLES.get(float(row["D"]), "solid"),
        )
    )
    return written


def load_runs_and_iterations(exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = exp_dir / "results"
    runs_path = results_dir / "runs.csv"
    iterations_path = results_dir / "iterations.csv"
    if not runs_path.exists():
        raise FileNotFoundError(runs_path)
    if not iterations_path.exists():
        raise FileNotFoundError(iterations_path)
    return pd.read_csv(runs_path), pd.read_csv(iterations_path)


def add_common_metadata(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["dataset_label"] = runs["dataset"].map(dataset_label) if "dataset" in runs else np.nan
    runs["method_label"] = runs["method"].map(method_label)
    return runs


def add_lasso_sklearn_references(runs: pd.DataFrame) -> pd.DataFrame:
    references = (
        runs[runs["method"] == "sklearn_saga"][
            [
                "dataset",
                "lambda",
                "seed",
                "train_loss",
                "test_loss",
                "test_accuracy",
                "nonzero_count",
                "final_norm",
                "runtime_seconds",
            ]
        ]
        .rename(
            columns={
                "train_loss": "sklearn_train_loss",
                "test_loss": "sklearn_test_loss",
                "test_accuracy": "sklearn_test_accuracy",
                "nonzero_count": "sklearn_nonzero_count",
                "final_norm": "sklearn_norm",
                "runtime_seconds": "sklearn_runtime_seconds",
            }
        )
        .drop_duplicates(["dataset", "lambda", "seed"])
    )
    if references.empty:
        raise ValueError("No sklearn_saga rows found in lasso runs.csv.")

    merged = runs.merge(
        references,
        on=["dataset", "lambda", "seed"],
        how="left",
        validate="many_to_one",
    )
    missing = merged["sklearn_train_loss"].isna()
    if missing.any():
        keys = (
            merged.loc[missing, ["dataset", "lambda", "seed"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Missing sklearn references for lasso keys: {keys}")

    merged["objective_gap_to_sklearn"] = (
        merged["train_loss"] - merged["sklearn_train_loss"]
    )
    merged.loc[merged["method"] == "sklearn_saga", "objective_gap_to_sklearn"] = 0.0
    return merged


def merge_iterations(
    iterations: pd.DataFrame,
    runs: pd.DataFrame,
    max_points: int,
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    run_ids = set(runs["run_id"])
    iterations = iterations[iterations["run_id"].isin(run_ids)].copy()
    if iterations.empty:
        return pd.DataFrame()
    iterations = iterations.sort_values(["run_id", "iteration"])
    if max_points > 0:
        iterations = pd.concat(
            [
                downsample_run(group, max_points)
                for _, group in iterations.groupby("run_id", sort=False)
            ],
            ignore_index=True,
        )
    metadata_cols = [
        col
        for col in runs.columns
        if col not in {"final_x", "final_x_hat", "final_parameter_vector"}
    ]
    return iterations.merge(
        runs[metadata_cols],
        on="run_id",
        how="inner",
        validate="many_to_one",
    )


def downsample_run(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(group) <= max_points:
        return group
    positions = np.linspace(0, len(group) - 1, max_points).round().astype(int)
    positions = np.unique(np.r_[0, positions, len(group) - 1])
    return group.iloc[positions]


def save_line_grid_figure(
    data: pd.DataFrame,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    title: str,
    group_col: str,
    y_col: str,
    y_label: str,
    label_func: Callable[[pd.Series], str],
    color_func: Callable[[pd.Series], str],
    style_func: Callable[[pd.Series], object],
    zero_line: bool = False,
    reference_func: Callable[[pd.DataFrame], float | None] | None = None,
    reference_label: str | None = None,
    force_log: bool = False,
    y_limits: tuple[float, float] | None = None,
    force_linear: bool = False,
    log_x: bool = False,
) -> list[Path]:
    if data.empty:
        print(f"Skipping {stem}: no rows match the requested slice.")
        return []

    groups = sorted(data[group_col].dropna().unique(), key=str)
    fig, axes = make_group_grid(groups)
    for index, group_value in enumerate(groups):
        ax = axes[index]
        subset = data[data[group_col] == group_value]
        y_values = plot_line_runs_on_ax(
            ax,
            subset,
            y_col=y_col,
            label_func=label_func,
            color_func=color_func,
            style_func=style_func,
        )
        if reference_func is not None:
            reference_y = reference_func(subset)
            if reference_y is not None and math.isfinite(reference_y):
                ax.axhline(
                    reference_y,
                    color="#111111",
                    linestyle="dashed",
                    linewidth=1.1,
                    alpha=0.7,
                    label=reference_label or "reference",
                )
                y_values.append(reference_y)
        finish_axes(
            ax,
            str(group_value),
            "Iteration",
            y_label,
            y_values,
            zero_line=zero_line,
            force_log=force_log,
            y_limits=y_limits,
            force_linear=force_linear,
            log_x=log_x,
        )

    hide_unused_axes(axes, len(groups))
    finalize_group_grid(fig, title)
    return save_figure(fig, out_dir, stem, formats)


def save_multi_metric_grid_figure(
    data: pd.DataFrame,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    title: str,
    group_col: str,
    metrics: list[str],
    y_label: str,
    label_func: Callable[[pd.Series, str], str],
    color_func: Callable[[pd.Series, str], str],
    style_func: Callable[[pd.Series, str], object],
    zero_line: bool = False,
    force_linear: bool = False,
    reference_func: Callable[[pd.DataFrame], float | None] | None = None,
    reference_label: str | None = None,
) -> list[Path]:
    if data.empty:
        print(f"Skipping {stem}: no rows match the requested slice.")
        return []

    groups = sorted(data[group_col].dropna().unique(), key=str)
    fig, axes = make_group_grid(groups)
    for index, group_value in enumerate(groups):
        ax = axes[index]
        subset = data[data[group_col] == group_value]
        y_values = plot_multi_metric_runs_on_ax(
            ax,
            subset,
            metrics=metrics,
            label_func=label_func,
            color_func=color_func,
            style_func=style_func,
        )
        if reference_func is not None:
            reference_y = reference_func(subset)
            if reference_y is not None and math.isfinite(reference_y):
                ax.axhline(
                    reference_y,
                    color="#111111",
                    linestyle="dashed",
                    linewidth=1.1,
                    alpha=0.7,
                    label=reference_label or "reference",
                )
                y_values.append(reference_y)
        finish_axes(
            ax,
            str(group_value),
            "Iteration",
            y_label,
            y_values,
            zero_line=zero_line,
            force_linear=force_linear,
        )

    hide_unused_axes(axes, len(groups))
    finalize_group_grid(fig, title)
    return save_figure(fig, out_dir, stem, formats)


def make_group_grid(groups: list[object]) -> tuple[plt.Figure, list[plt.Axes]]:
    n_cols = min(3, max(len(groups), 1))
    n_rows = math.ceil(max(len(groups), 1) / n_cols)
    fig, axes_array = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.15 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    return fig, list(axes_array.ravel())


def hide_unused_axes(axes: list[plt.Axes], used_count: int) -> None:
    for ax in axes[used_count:]:
        ax.set_visible(False)


def finalize_group_grid(fig: plt.Figure, title: str) -> None:
    handles, labels = collect_legend_items(fig)
    if handles:
        ncol = min(4, max(1, len(handles)))
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=ncol,
            frameon=False,
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.965))


def single_reference(column: str) -> Callable[[pd.DataFrame], float | None]:
    def reference(data: pd.DataFrame) -> float | None:
        values = data[[column]].dropna().drop_duplicates()
        if len(values) != 1:
            return None
        return float(values.iloc[0][column])

    return reference


def plot_line_runs_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    y_col: str,
    label_func: Callable[[pd.Series], str],
    color_func: Callable[[pd.Series], str],
    style_func: Callable[[pd.Series], object],
) -> list[float]:
    y_values: list[float] = []
    for _, run_data in data.groupby("run_id", sort=False):
        run_data = run_data.sort_values("iteration")
        first = run_data.iloc[0]
        series = run_data[["iteration", y_col]].dropna()
        if series.empty:
            continue
        ax.plot(
            series["iteration"],
            series[y_col],
            label=label_func(first),
            color=color_func(first),
            linestyle=style_func(first),
            alpha=0.9,
        )
        y_values.extend(series[y_col].to_numpy(dtype=float))
    return y_values


def plot_multi_metric_runs_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    metrics: list[str],
    label_func: Callable[[pd.Series, str], str],
    color_func: Callable[[pd.Series, str], str],
    style_func: Callable[[pd.Series, str], object],
) -> list[float]:
    y_values: list[float] = []
    for _, run_data in data.groupby("run_id", sort=False):
        run_data = run_data.sort_values("iteration")
        first = run_data.iloc[0]
        for metric in metrics:
            if metric not in run_data:
                continue
            series = run_data[["iteration", metric]].dropna()
            if series.empty:
                continue
            ax.plot(
                series["iteration"],
                series[metric],
                label=label_func(first, metric),
                color=color_func(first, metric),
                linestyle=style_func(first, metric),
                alpha=0.9,
            )
            y_values.extend(series[metric].to_numpy(dtype=float))
    return y_values


def save_line_figure(
    data: pd.DataFrame,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    title: str,
    y_col: str,
    y_label: str,
    label_func: Callable[[pd.Series], str],
    color_func: Callable[[pd.Series], str],
    style_func: Callable[[pd.Series], object],
    zero_line: bool = False,
) -> list[Path]:
    if data.empty:
        print(f"Skipping {stem}: no rows match the requested slice.")
        return []

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    y_values: list[float] = []
    for _, run_data in data.groupby("run_id", sort=False):
        run_data = run_data.sort_values("iteration")
        first = run_data.iloc[0]
        series = run_data[["iteration", y_col]].dropna()
        if series.empty:
            continue
        ax.plot(
            series["iteration"],
            series[y_col],
            label=label_func(first),
            color=color_func(first),
            linestyle=style_func(first),
            alpha=0.9,
        )
        y_values.extend(series[y_col].to_numpy(dtype=float))

    finish_axes(ax, title, "Iteration", y_label, y_values, zero_line=zero_line)
    place_legend(ax)
    return save_figure(fig, out_dir, stem, formats)


def save_multi_metric_figure(
    data: pd.DataFrame,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    title: str,
    metrics: list[str],
    y_label: str,
    label_func: Callable[[pd.Series, str], str],
    color_func: Callable[[pd.Series, str], str],
    style_func: Callable[[pd.Series, str], object],
    reference_y: float | None = None,
    reference_label: str | None = None,
    force_linear: bool = False,
) -> list[Path]:
    if data.empty:
        print(f"Skipping {stem}: no rows match the requested slice.")
        return []

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    y_values: list[float] = []
    for _, run_data in data.groupby("run_id", sort=False):
        run_data = run_data.sort_values("iteration")
        first = run_data.iloc[0]
        for metric in metrics:
            if metric not in run_data:
                continue
            series = run_data[["iteration", metric]].dropna()
            if series.empty:
                continue
            ax.plot(
                series["iteration"],
                series[metric],
                label=label_func(first, metric),
                color=color_func(first, metric),
                linestyle=style_func(first, metric),
                alpha=0.9,
            )
            y_values.extend(series[metric].to_numpy(dtype=float))

    if reference_y is not None and math.isfinite(reference_y):
        ax.axhline(
            reference_y,
            color="#111111",
            linestyle="dashed",
            linewidth=1.1,
            alpha=0.7,
            label=reference_label or "reference",
        )
        y_values.append(reference_y)

    finish_axes(
        ax,
        title,
        "Iteration",
        y_label,
        y_values,
        force_linear=force_linear,
    )
    place_legend(ax)
    return save_figure(fig, out_dir, stem, formats)


def save_runtime_gap_grid(
    runs: pd.DataFrame,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    group_col: str,
    y_col: str,
    title: str,
) -> list[Path]:
    data = runs.dropna(subset=["runtime_seconds", y_col, group_col]).copy()
    if data.empty:
        print(f"Skipping {stem}: no rows have runtime and final gap values.")
        return []

    groups = sorted(data[group_col].dropna().unique(), key=str)
    n_cols = min(3, len(groups))
    n_rows = math.ceil(len(groups) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.65 * n_cols, 2.95 * n_rows),
        squeeze=False,
    )

    y_values = data[y_col].to_numpy(dtype=float)
    for index, group_value in enumerate(groups):
        ax = axes[index // n_cols][index % n_cols]
        subset = data[data[group_col] == group_value]
        for method in sorted(subset["method"].dropna().unique(), key=str):
            method_data = subset[subset["method"] == method]
            ax.scatter(
                method_data["runtime_seconds"],
                method_data[y_col],
                label=method_label(method),
                color=METHOD_COLORS.get(str(method), "#555555"),
                marker=marker_by_method(method),
                s=20,
                alpha=0.78,
                edgecolor="none",
            )
        ax.set_title(str(group_value))
        ax.set_xscale("log")
        set_runtime_x_ticks(ax, subset["runtime_seconds"].to_numpy(dtype=float))
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Final objective gap")
        apply_y_scale(ax, y_values, zero_line=True)
        clean_axes(ax)

    for index in range(len(groups), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_visible(False)

    handles, labels = collect_legend_items(fig)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(4, len(handles)),
            frameon=False,
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    return save_figure(fig, out_dir, stem, formats, close=True)


def set_runtime_x_ticks(ax: plt.Axes, values: np.ndarray) -> None:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return
    x_min = float(finite.min())
    x_max = float(finite.max())
    if x_min == x_max:
        x_min *= 0.8
        x_max *= 1.25
    else:
        padding = 0.12
        log_min = math.log10(x_min)
        log_max = math.log10(x_max)
        span = log_max - log_min
        x_min = 10.0 ** (log_min - padding * span)
        x_max = 10.0 ** (log_max + padding * span)

    ticks = np.geomspace(x_min, x_max, num=6)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.2g}"))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=36)
    )


def finish_axes(
    ax: plt.Axes,
    title: str,
    x_label: str,
    y_label: str,
    y_values: list[float],
    *,
    zero_line: bool = False,
    force_linear: bool = False,
    force_log: bool = False,
    y_limits: tuple[float, float] | None = None,
    log_x: bool = False,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale("log")
    if force_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
        )
    elif force_linear:
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    else:
        apply_y_scale(ax, np.asarray(y_values, dtype=float), zero_line=zero_line)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if zero_line and ax.get_yscale() != "log":
        ax.axhline(0.0, color="#111111", linestyle="dashed", linewidth=0.9, alpha=0.55)
    clean_axes(ax)


def apply_y_scale(ax: plt.Axes, values: np.ndarray, *, zero_line: bool = False) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    if np.all(finite > 0):
        ax.set_yscale("log")
        return
    if finite.min() < 0 < finite.max() or zero_line:
        positive_abs = np.abs(finite[finite != 0])
        linthresh = float(np.nanpercentile(positive_abs, 5)) if positive_abs.size else 1e-6
        ax.set_yscale("symlog", linthresh=max(linthresh, 1e-8))


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", alpha=0.45)
    ax.grid(True, which="minor", alpha=0.18)


def place_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    if not dedup:
        return
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )


def collect_legend_items(fig: plt.Figure) -> tuple[list[object], list[str]]:
    dedup: dict[str, object] = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            dedup.setdefault(label, handle)
    return list(dedup.values()), list(dedup.keys())


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    close: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt_name in formats:
        path = out_dir / f"{stem}.{fmt_name}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    if close:
        plt.close(fig)
    return written


def dataset_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return Path(str(value)).name


def method_label(method: object) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def marker_by_method(method: object) -> str:
    keys = sorted(METHOD_LABELS)
    try:
        return MARKERS[keys.index(str(method)) % len(MARKERS)]
    except ValueError:
        return "o"


def tuning_label(row: pd.Series) -> str:
    method = method_label(row["method"])
    if pd.notna(row.get("alpha")):
        return f"{method}, alpha={fmt(row['alpha'])}"
    return f"{method}, gamma_mult={fmt(row['gamma_mult'])}"


def tuning_style(row: pd.Series) -> object:
    value = row.get("alpha")
    if pd.isna(value):
        value = row.get("gamma_mult")
    styles = ["solid", "dashed", "dotted", "dashdot", (0, (5, 1)), (0, (1, 1))]
    return styles[stable_index(fmt(value), len(styles))]


def prox_label(row: pd.Series) -> str:
    profile = row.get("prox_weights_profile")
    if pd.isna(profile) or str(profile) == "none":
        profile = "none"
    return f"{method_label(row['method'])}, {profile}"


def prox_color(row: pd.Series) -> str:
    if str(row.get("prox_fun")) == "weighted_euclidean":
        return "#ff7f0e" if str(row.get("dual_averaging")) == "weighted" else "#9467bd"
    return METHOD_COLORS.get(str(row["method"]), "#555555")


def prox_style(row: pd.Series) -> object:
    profile = str(row.get("prox_weights_profile"))
    if profile == "coordinate_scale":
        return "solid"
    if profile == "inverse_coordinate_scale":
        return "dashed"
    if profile == "uniform":
        return "dotted"
    return "dashdot" if str(row.get("dual_averaging")) == "weighted" else "solid"


def color_by_value(value: object) -> str:
    text = fmt(value)
    return PALETTE[stable_index(text, len(PALETTE))]


def gamma_color(value: object) -> str:
    numeric = float(value)
    return GAMMA_COLORS.get(numeric, color_by_value(value))


def stable_index(text: str, modulo: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def near(series: pd.Series, value: float) -> pd.Series:
    return np.isclose(pd.to_numeric(series, errors="coerce"), value, rtol=0.0, atol=1e-12)


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def slug(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_").lower()


if __name__ == "__main__":
    main()
