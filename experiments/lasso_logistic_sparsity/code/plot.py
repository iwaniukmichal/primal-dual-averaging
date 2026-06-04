from __future__ import annotations

import argparse
import math
import webbrowser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_OUT_DIR = EXPERIMENT_DIR / "plots"

PLOT_CHOICES = {
    "trajectories",
    "objective-gaps",
    "norms",
    "gradients",
    "sparsity",
    "aggregates",
    "all",
}
TRAJECTORY_METRICS = {
    "objective_value_x_hat",
    "objective_value_x",
    "objective_gap_to_sklearn_x_hat",
    "objective_gap_to_sklearn_x",
    "normalized_gap",
    "x_hat_norm",
    "x_norm",
    "g_norm",
    "nonzero_count",
}
METHOD_ORDER = [
    "sda_simple_euclidean",
    "sda_weighted_euclidean",
    "ssa_euclidean",
    "projected_subgradient",
    "sklearn_saga",
]
METHOD_LABELS = {
    "sda_simple_euclidean": "SDA simple",
    "sda_weighted_euclidean": "SDA weighted",
    "ssa_euclidean": "SSA",
    "projected_subgradient": "Projected subgradient",
    "sklearn_saga": "sklearn saga",
}
METHOD_COLORS = {
    "sda_simple_euclidean": "#2563eb",
    "sda_weighted_euclidean": "#dc2626",
    "ssa_euclidean": "#7c3aed",
    "projected_subgradient": "#16a34a",
    "sklearn_saga": "#111827",
}
DASH_BY_D = {
    8.0: "solid",
    32.0: "dash",
    128.0: "dot",
}
ITERATION_COLUMNS = [
    "run_id",
    "iteration",
    "objective_value_x",
    "objective_value_x_hat",
    "normalized_gap",
    "x_norm",
    "x_hat_norm",
    "g_norm",
    "nonzero_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Plotly diagnostics for lasso_logistic_sparsity results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing iterations.csv, runs.csv, and summary.csv. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory where HTML plots are written. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset basename, objective_id, or full path to plot. "
            "Repeat to include multiple datasets. Default: all datasets."
        ),
    )
    parser.add_argument(
        "--lambda",
        dest="lambdas",
        action="append",
        type=float,
        default=[],
        help="L1 lambda value to include. Repeat to include multiple values. Default: all lambdas.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Seed to include. Repeat to include multiple seeds. Default: all seeds.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(TRAJECTORY_METRICS),
        default="objective_value_x_hat",
        help="Primary metric for train objective trajectory plots. Default: objective_value_x_hat.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=sorted(PLOT_CHOICES),
        default=["all"],
        help="Plot groups to write. Default: all.",
    )
    parser.add_argument(
        "--max-traces-per-dataset",
        type=int,
        default=900,
        help="Skip trajectory figures that would exceed this trace count. Default: 900.",
    )
    parser.add_argument(
        "--max-points-per-run",
        type=int,
        default=300,
        help=(
            "Maximum points from each run in trajectory plots. "
            "Use 0 to keep every iteration. Default: 300."
        ),
    )
    parser.add_argument(
        "--iteration-chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk while reading iterations.csv. Default: 200000.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open generated HTML files in the default browser.",
    )
    parser.add_argument(
        "--show-traces",
        action="store_true",
        help="Show trajectory traces by default. Without this, traces start hidden and can be enabled from the legend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_plots = expand_plot_choices(args.plots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs, summary = load_run_results(args.results_dir)
    runs = add_reference_metadata(add_plot_metadata(runs))
    runs = filter_runs(runs, datasets=args.dataset, lambdas=args.lambdas, seeds=args.seed)
    summary = filter_summary(summary, runs)
    if runs.empty:
        raise ValueError("No runs remain after filtering.")

    written: list[Path] = []
    if "aggregates" in selected_plots:
        written.extend(write_aggregate_figures(runs, summary, args.out_dir))

    trajectory_groups = selected_plots - {"aggregates"}
    if trajectory_groups:
        custom_runs = runs[runs["method"] != "sklearn_saga"].copy()
        iteration_data = load_iteration_data(
            args.results_dir / "iterations.csv",
            run_ids=set(custom_runs["run_id"]),
            metadata=custom_runs[run_metadata_columns()],
            chunksize=args.iteration_chunksize,
            max_points_per_run=args.max_points_per_run,
        )
        for dataset_label in sorted(custom_runs["dataset_label"].dropna().unique()):
            dataset_iterations = iteration_data[
                iteration_data["dataset_label"] == dataset_label
            ].copy()
            if dataset_iterations.empty:
                continue

            if "trajectories" in trajectory_groups:
                written.append(
                    write_objective_trajectory_figure(
                        dataset_label,
                        dataset_iterations,
                        args.metric,
                        args.out_dir,
                        args.max_traces_per_dataset,
                        visible_by_default=args.show_traces,
                    )
                )
            if "objective-gaps" in trajectory_groups:
                written.append(
                    write_objective_gap_figure(
                        dataset_label,
                        dataset_iterations,
                        args.out_dir,
                        args.max_traces_per_dataset,
                        visible_by_default=args.show_traces,
                    )
                )
            if "norms" in trajectory_groups:
                written.append(
                    write_norm_figure(
                        dataset_label,
                        dataset_iterations,
                        args.out_dir,
                        args.max_traces_per_dataset,
                        visible_by_default=args.show_traces,
                    )
                )
            if "gradients" in trajectory_groups:
                written.append(
                    write_gradient_figure(
                        dataset_label,
                        dataset_iterations,
                        args.out_dir,
                        args.max_traces_per_dataset,
                        visible_by_default=args.show_traces,
                    )
                )
            if "sparsity" in trajectory_groups:
                written.append(
                    write_sparsity_figure(
                        dataset_label,
                        dataset_iterations,
                        args.out_dir,
                        args.max_traces_per_dataset,
                        visible_by_default=args.show_traces,
                    )
                )

    print(f"Wrote {len(written)} Plotly HTML file(s) to {args.out_dir}")
    for path in written:
        print(path)
    if args.show:
        for path in written:
            webbrowser.open(path.resolve().as_uri())


def expand_plot_choices(values: Iterable[str]) -> set[str]:
    requested = set(values)
    if "all" in requested:
        return {
            "trajectories",
            "objective-gaps",
            "norms",
            "gradients",
            "sparsity",
            "aggregates",
        }
    return requested


def load_run_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_path = results_dir / "runs.csv"
    summary_path = results_dir / "summary.csv"
    for path in (runs_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    runs = pd.read_csv(runs_path)
    summary = pd.read_csv(summary_path)
    require_columns(
        runs,
        {
            "run_id",
            "method",
            "objective_id",
            "dataset",
            "D",
            "gamma_mult",
            "alpha",
            "lambda",
            "seed",
            "iterations",
            "converged",
            "runtime_seconds",
            "train_loss",
            "test_loss",
            "test_accuracy",
            "nonzero_count",
            "final_norm",
            "final_gap",
        },
        runs_path,
    )
    require_columns(
        summary,
        {
            "method",
            "objective_id",
            "dataset",
            "runs",
            "mean_runtime_seconds",
            "mean_final_gap",
            "mean_test_accuracy",
        },
        summary_path,
    )
    return runs, summary


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def add_plot_metadata(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["dataset_label"] = runs["dataset"].map(dataset_label)
    runs["method_label"] = runs["method"].map(METHOD_LABELS).fillna(runs["method"])
    runs["tuning_name"] = np.where(runs["alpha"].notna(), "alpha", "gamma_mult")
    runs["tuning_value"] = runs["alpha"].where(runs["alpha"].notna(), runs["gamma_mult"])
    runs["tuning_label"] = runs.apply(
        lambda row: (
            "sklearn"
            if row["method"] == "sklearn_saga"
            else f"{row['tuning_name']}={format_numeric(row['tuning_value'])}"
        ),
        axis=1,
    )
    runs["D_label"] = runs["D"].map(
        lambda value: "" if pd.isna(value) else f"D={format_numeric(value)}"
    )
    runs["config_label"] = runs.apply(config_label, axis=1)
    return runs


def add_reference_metadata(runs: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError("No sklearn_saga rows found; cannot build empirical optimum references.")

    runs = runs.merge(references, on=["dataset", "lambda", "seed"], how="left", validate="many_to_one")
    missing = runs["sklearn_train_loss"].isna()
    if missing.any():
        missing_keys = (
            runs.loc[missing, ["dataset_label", "lambda", "seed"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Missing sklearn_saga reference for run key(s): {missing_keys}")

    runs["objective_gap_to_sklearn"] = runs["train_loss"] - runs["sklearn_train_loss"]
    runs.loc[runs["method"] == "sklearn_saga", "objective_gap_to_sklearn"] = 0.0
    runs["test_loss_gap_to_sklearn"] = runs["test_loss"] - runs["sklearn_test_loss"]
    runs["norm_gap_to_sklearn"] = runs["final_norm"] - runs["sklearn_norm"]
    runs["nonzero_gap_to_sklearn"] = runs["nonzero_count"] - runs["sklearn_nonzero_count"]
    return runs


def filter_runs(
    runs: pd.DataFrame,
    *,
    datasets: list[str],
    lambdas: list[float],
    seeds: list[int],
) -> pd.DataFrame:
    filtered = runs.copy()
    if datasets:
        requested = set(datasets)
        dataset_mask = (
            filtered["dataset_label"].isin(requested)
            | filtered["objective_id"].isin(requested)
            | filtered["dataset"].isin(requested)
        )
        missing = sorted(
            requested
            - set(filtered.loc[dataset_mask, "dataset_label"])
            - set(filtered.loc[dataset_mask, "objective_id"])
            - set(filtered.loc[dataset_mask, "dataset"])
        )
        if missing:
            raise ValueError(f"Unknown dataset filter(s): {', '.join(missing)}")
        filtered = filtered[dataset_mask].copy()
    if lambdas:
        filtered = filtered[filtered["lambda"].isin(lambdas)].copy()
    if seeds:
        filtered = filtered[filtered["seed"].isin(seeds)].copy()
    return filtered


def filter_summary(summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    keys = runs[["method", "objective_id", "dataset"]].drop_duplicates()
    summary = summary.merge(keys, on=["method", "objective_id", "dataset"], how="inner")
    summary = summary.copy()
    summary["dataset_label"] = summary["dataset"].map(dataset_label)
    summary["method_label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    return summary


def run_metadata_columns() -> list[str]:
    return [
        "run_id",
        "method",
        "method_label",
        "objective_id",
        "dataset",
        "dataset_label",
        "D",
        "D_label",
        "gamma_mult",
        "alpha",
        "lambda",
        "seed",
        "tuning_label",
        "config_label",
        "iterations",
        "converged",
        "runtime_seconds",
        "train_loss",
        "test_loss",
        "test_accuracy",
        "final_norm",
        "final_gap",
        "sklearn_train_loss",
        "sklearn_norm",
        "sklearn_nonzero_count",
    ]


def load_iteration_data(
    iterations_path: Path,
    *,
    run_ids: set[str],
    metadata: pd.DataFrame,
    chunksize: int,
    max_points_per_run: int,
) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame()
    if not iterations_path.exists():
        raise FileNotFoundError(iterations_path)

    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(iterations_path, usecols=ITERATION_COLUMNS, chunksize=chunksize):
        chunk = chunk[chunk["run_id"].isin(run_ids)]
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        return pd.DataFrame()

    iterations = pd.concat(pieces, ignore_index=True)
    iterations = iterations.sort_values(["run_id", "iteration"])
    iterations = pd.concat(
        [
            downsample_run(group, max_points_per_run)
            for _, group in iterations.groupby("run_id", sort=False)
        ],
        ignore_index=True,
    )

    data = iterations.merge(metadata, on="run_id", how="inner", validate="many_to_one")
    data["objective_gap_to_sklearn_x"] = data["objective_value_x"] - data["sklearn_train_loss"]
    data["objective_gap_to_sklearn_x_hat"] = (
        data["objective_value_x_hat"] - data["sklearn_train_loss"]
    )
    return data


def write_objective_trajectory_figure(
    dataset: str,
    data: pd.DataFrame,
    metric: str,
    out_dir: Path,
    max_traces: int,
    *,
    visible_by_default: bool,
) -> Path:
    metrics = [metric]
    if metric == "objective_value_x_hat":
        metrics.append("objective_value_x")
    elif metric == "objective_value_x":
        metrics.append("objective_value_x_hat")
    metrics = [column for column in dict.fromkeys(metrics) if column in data]
    is_objective_value_plot = all(metric.startswith("objective_value_") for metric in metrics)
    fig = trajectory_figure(
        data,
        metrics=metrics,
        title=f"{dataset}: train objective trajectories",
        y_axis_title="Train objective" if is_objective_value_plot else metric_label(metrics[0]),
        max_traces=max_traces,
        visible_by_default=visible_by_default,
    )
    if is_objective_value_plot:
        add_reference_lines(
            fig,
            data,
            value_column="sklearn_train_loss",
            label_prefix="sklearn objective",
        )
    elif any(metric.startswith("objective_gap_to_sklearn_") for metric in metrics):
        fig.add_hline(y=0.0, line_dash="dash", line_color="#111827", opacity=0.65)
    path = out_dir / f"trajectories_{slug(dataset)}.html"
    write_figure(fig, path)
    return path


def write_objective_gap_figure(
    dataset: str,
    data: pd.DataFrame,
    out_dir: Path,
    max_traces: int,
    *,
    visible_by_default: bool,
) -> Path:
    fig = trajectory_figure(
        data,
        metrics=["objective_gap_to_sklearn_x_hat"],
        title=f"{dataset}: train objective gap to sklearn",
        y_axis_title="Train objective gap to sklearn",
        max_traces=max_traces,
        visible_by_default=visible_by_default,
        log_y=None,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#111827", opacity=0.65)
    path = out_dir / f"objective_gaps_to_sklearn_{slug(dataset)}.html"
    write_figure(fig, path)
    return path


def write_norm_figure(
    dataset: str,
    data: pd.DataFrame,
    out_dir: Path,
    max_traces: int,
    *,
    visible_by_default: bool,
) -> Path:
    fig = trajectory_figure(
        data,
        metrics=["x_norm", "x_hat_norm"],
        title=f"{dataset}: iterate norm trajectories",
        y_axis_title="Euclidean norm",
        max_traces=max_traces,
        visible_by_default=visible_by_default,
        log_y=False,
    )
    add_reference_lines(
        fig,
        data,
        value_column="sklearn_norm",
        label_prefix="||x_sklearn||",
    )
    path = out_dir / f"norms_{slug(dataset)}.html"
    write_figure(fig, path)
    return path


def write_gradient_figure(
    dataset: str,
    data: pd.DataFrame,
    out_dir: Path,
    max_traces: int,
    *,
    visible_by_default: bool,
) -> Path:
    fig = trajectory_figure(
        data,
        metrics=["g_norm"],
        title=f"{dataset}: subgradient norm trajectories",
        y_axis_title="Subgradient norm",
        max_traces=max_traces,
        visible_by_default=visible_by_default,
    )
    path = out_dir / f"gradients_{slug(dataset)}.html"
    write_figure(fig, path)
    return path


def write_sparsity_figure(
    dataset: str,
    data: pd.DataFrame,
    out_dir: Path,
    max_traces: int,
    *,
    visible_by_default: bool,
) -> Path:
    fig = trajectory_figure(
        data,
        metrics=["nonzero_count"],
        title=f"{dataset}: sparsity trajectories",
        y_axis_title="Nonzero parameter count",
        max_traces=max_traces,
        visible_by_default=visible_by_default,
        log_y=False,
    )
    add_reference_lines(
        fig,
        data,
        value_column="sklearn_nonzero_count",
        label_prefix="sklearn nonzeros",
    )
    path = out_dir / f"sparsity_{slug(dataset)}.html"
    write_figure(fig, path)
    return path


def trajectory_figure(
    data: pd.DataFrame,
    *,
    metrics: list[str],
    title: str,
    y_axis_title: str,
    max_traces: int,
    visible_by_default: bool,
    log_y: bool | None = None,
) -> go.Figure:
    trace_count = int(data["run_id"].nunique()) * len(metrics)
    if trace_count > max_traces:
        raise ValueError(
            f"{title} would create {trace_count} traces, exceeding --max-traces-per-dataset={max_traces}."
        )

    fig = go.Figure()
    metric_dash = metric_dash_lookup(metrics)
    for method in methods_present(data):
        method_data = data[data["method"] == method]
        for run_id, run_data in method_data.groupby("run_id", sort=False):
            run_data = run_data.sort_values("iteration")
            first = run_data.iloc[0]
            for metric in metrics:
                series = run_data[["iteration", metric]].dropna()
                if series.empty:
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=series["iteration"],
                        y=series[metric],
                        mode="lines+markers",
                        name=f"{first['config_label']} | {metric}",
                        legendgroup=f"{run_id}::{metric}",
                        showlegend=True,
                        visible=True if visible_by_default else "legendonly",
                        line={
                            "color": METHOD_COLORS.get(str(first["method"]), "#4b5563"),
                            "dash": dash_for_run(first, metric_dash[metric]),
                            "width": 1.8,
                        },
                        marker={
                            "color": METHOD_COLORS.get(str(first["method"]), "#4b5563"),
                            "size": 4,
                            "opacity": 0.85,
                        },
                        meta=[
                            first["method_label"],
                            first["tuning_label"],
                            first["D_label"],
                            float(first["lambda"]),
                            int(first["seed"]),
                            bool(first["converged"]),
                            float(first["runtime_seconds"]),
                            metric,
                        ],
                        hovertemplate=(
                            "%{meta[0]}<br>"
                            "%{meta[1]}<br>"
                            "%{meta[2]}<br>"
                            "lambda=%{meta[3]:.6g}<br>"
                            "seed=%{meta[4]}<br>"
                            "metric=%{meta[7]}<br>"
                            "iteration=%{x}<br>"
                            "value=%{y:.6g}<br>"
                            "converged=%{meta[5]}<br>"
                            "runtime=%{meta[6]:.6g}s"
                            "<extra></extra>"
                        ),
                    )
                )

    use_log_y = can_use_log_scale(data, metrics) if log_y is None else log_y
    fig.update_xaxes(title_text="Iteration", range=axis_range(data["iteration"], log_scale=False))
    fig.update_yaxes(
        title_text=y_axis_title,
        type="log" if use_log_y else "linear",
        range=axis_range(data[metrics].stack().dropna(), log_scale=use_log_y),
        zeroline=False,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        height=780,
        legend_title="Configuration",
        legend={
            "itemsizing": "constant",
            "traceorder": "normal",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
        },
        margin={"l": 80, "r": 420, "t": 80, "b": 60},
    )
    return fig


def add_reference_lines(
    fig: go.Figure,
    data: pd.DataFrame,
    *,
    value_column: str,
    label_prefix: str,
) -> None:
    references = (
        data[["lambda", "seed", value_column]]
        .dropna()
        .drop_duplicates()
        .sort_values(["lambda", "seed"])
    )
    for index, row in references.iterrows():
        show_label = index == references.index[0]
        line_options = {
            "y": float(row[value_column]),
            "line_dash": "dash",
            "line_color": "#111827",
            "opacity": 0.22,
        }
        if show_label:
            line_options["annotation_text"] = (
                f"{label_prefix}={format_numeric(row[value_column])}"
            )
            line_options["annotation_position"] = "top left"
        fig.add_hline(**line_options)


def write_aggregate_figures(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    return [
        write_runtime_objective_gap_scatter(runs, out_dir),
        write_accuracy_sparsity_scatter(runs, out_dir),
        write_test_loss_sparsity_scatter(runs, out_dir),
        write_runtime_accuracy_scatter(runs, out_dir),
        write_summary_comparison(summary, out_dir),
    ]


def write_runtime_objective_gap_scatter(runs: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.scatter(
        runs,
        x="runtime_seconds",
        y="objective_gap_to_sklearn",
        color="method_label",
        symbol="method_label",
        facet_col="dataset_label",
        facet_col_wrap=3,
        hover_data=[
            "lambda",
            "seed",
            "D",
            "gamma_mult",
            "alpha",
            "iterations",
            "train_loss",
            "sklearn_train_loss",
            "test_accuracy",
            "nonzero_count",
            "final_norm",
        ],
        title="Runtime vs final objective gap to sklearn",
        labels={
            "runtime_seconds": "Runtime (seconds)",
            "objective_gap_to_sklearn": "Final train objective gap to sklearn",
            "method_label": "Method",
            "dataset_label": "Dataset",
        },
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#111827", opacity=0.55)
    fig.update_layout(template="plotly_white", height=950)
    path = out_dir / "aggregate_runtime_vs_final_objective_gap.html"
    write_figure(fig, path)
    return path


def write_accuracy_sparsity_scatter(runs: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.scatter(
        runs,
        x="nonzero_count",
        y="test_accuracy",
        color="method_label",
        symbol="lambda",
        facet_col="dataset_label",
        facet_col_wrap=3,
        hover_data=[
            "seed",
            "D",
            "gamma_mult",
            "alpha",
            "runtime_seconds",
            "test_loss",
            "objective_gap_to_sklearn",
        ],
        title="Test accuracy vs sparsity",
        labels={
            "nonzero_count": "Nonzero parameter count",
            "test_accuracy": "Test accuracy",
            "method_label": "Method",
            "dataset_label": "Dataset",
        },
    )
    fig.update_layout(template="plotly_white", height=950)
    path = out_dir / "aggregate_accuracy_vs_sparsity.html"
    write_figure(fig, path)
    return path


def write_test_loss_sparsity_scatter(runs: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.scatter(
        runs,
        x="nonzero_count",
        y="test_loss",
        color="method_label",
        symbol="lambda",
        facet_col="dataset_label",
        facet_col_wrap=3,
        hover_data=[
            "seed",
            "D",
            "gamma_mult",
            "alpha",
            "runtime_seconds",
            "test_accuracy",
            "objective_gap_to_sklearn",
        ],
        title="Test loss vs sparsity",
        labels={
            "nonzero_count": "Nonzero parameter count",
            "test_loss": "Test loss",
            "method_label": "Method",
            "dataset_label": "Dataset",
        },
    )
    fig.update_layout(template="plotly_white", height=950)
    path = out_dir / "aggregate_test_loss_vs_sparsity.html"
    write_figure(fig, path)
    return path


def write_runtime_accuracy_scatter(runs: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.scatter(
        runs,
        x="runtime_seconds",
        y="test_accuracy",
        color="method_label",
        symbol="lambda",
        facet_col="dataset_label",
        facet_col_wrap=3,
        hover_data=[
            "seed",
            "D",
            "gamma_mult",
            "alpha",
            "iterations",
            "test_loss",
            "nonzero_count",
            "objective_gap_to_sklearn",
        ],
        title="Runtime vs test accuracy",
        labels={
            "runtime_seconds": "Runtime (seconds)",
            "test_accuracy": "Test accuracy",
            "method_label": "Method",
            "dataset_label": "Dataset",
        },
    )
    fig.update_layout(template="plotly_white", height=950)
    path = out_dir / "aggregate_runtime_vs_test_accuracy.html"
    write_figure(fig, path)
    return path


def write_summary_comparison(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.bar(
        summary,
        x="dataset_label",
        y="mean_test_accuracy",
        color="method_label",
        barmode="group",
        hover_data=["runs", "mean_runtime_seconds", "mean_final_gap"],
        title="Mean test accuracy by dataset and method",
        labels={
            "dataset_label": "Dataset",
            "mean_test_accuracy": "Mean test accuracy",
            "method_label": "Method",
        },
    )
    fig.update_layout(template="plotly_white", height=700)
    path = out_dir / "aggregate_summary_accuracy.html"
    write_figure(fig, path)
    return path


def methods_present(data: pd.DataFrame) -> list[str]:
    present = set(data["method"].dropna().unique())
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def metric_label(metric: str) -> str:
    labels = {
        "objective_value_x": "Train objective at x",
        "objective_value_x_hat": "Train objective at x_hat",
        "objective_gap_to_sklearn_x": "Train objective gap to sklearn at x",
        "objective_gap_to_sklearn_x_hat": "Train objective gap to sklearn at x_hat",
        "normalized_gap": "Normalized SDA certificate gap",
        "x_norm": "x norm",
        "x_hat_norm": "x_hat norm",
        "g_norm": "Subgradient norm",
        "nonzero_count": "Nonzero parameter count",
    }
    return labels.get(metric, metric.replace("_", " "))


def metric_dash_lookup(metrics: list[str]) -> dict[str, str]:
    dash_sequence = ["solid", "dash", "dot", "dashdot"]
    return {
        metric: dash_sequence[index % len(dash_sequence)]
        for index, metric in enumerate(metrics)
    }


def dash_for_run(row: pd.Series, metric_dash: str) -> str:
    if metric_dash != "solid":
        return metric_dash
    if pd.isna(row["D"]):
        return "solid"
    return DASH_BY_D.get(float(row["D"]), "solid")


def can_use_log_scale(data: pd.DataFrame, metrics: list[str]) -> bool:
    values = data[metrics].stack().dropna()
    return not values.empty and bool((values > 0).all())


def axis_range(values: pd.Series, *, log_scale: bool) -> list[float] | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    if log_scale:
        numeric = numeric[numeric > 0]
        if numeric.empty:
            return None
        min_value = float(numeric.min())
        max_value = float(numeric.max())
        if min_value == max_value:
            min_value *= 0.9
            max_value *= 1.1
        return [math.log10(min_value), math.log10(max_value)]

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if min_value == max_value:
        padding = abs(min_value) * 0.05 or 0.05
    else:
        padding = (max_value - min_value) * 0.03
    return [min_value - padding, max_value + padding]


def downsample_run(run_data: pd.DataFrame, max_points_per_run: int) -> pd.DataFrame:
    if max_points_per_run <= 0 or len(run_data) <= max_points_per_run:
        return run_data
    positions = np.linspace(0, len(run_data) - 1, num=max_points_per_run, dtype=int)
    positions = np.unique(positions)
    return run_data.iloc[positions]


def dataset_label(value: object) -> str:
    text = str(value)
    return Path(text).name if text else ""


def config_label(row: pd.Series) -> str:
    if row["method"] == "sklearn_saga":
        return f"{row['method_label']} | lambda={format_numeric(row['lambda'])} | seed={int(row['seed'])}"
    d_label = f" | {row['D_label']}" if row["D_label"] else ""
    return (
        f"{row['method_label']} | {row['tuning_label']}{d_label} | "
        f"lambda={format_numeric(row['lambda'])} | seed={int(row['seed'])}"
    )


def format_numeric(value: object) -> str:
    if pd.isna(value):
        return "NA"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def slug(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def write_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


if __name__ == "__main__":
    main()
