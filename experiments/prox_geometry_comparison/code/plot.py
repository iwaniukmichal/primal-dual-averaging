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

PLOT_CHOICES = {"trajectories", "runtime-gap", "all"}
TRAJECTORY_METRICS = {
    "objective_gap_x_hat",
    "objective_gap_x",
    "objective_value_x_hat",
    "objective_value_x",
    "normalized_gap",
}
METHOD_ORDER = [
    "sda_simple_euclidean",
    "sda_simple_weighted_euclidean",
    "sda_weighted_euclidean",
    "sda_weighted_weighted_euclidean",
]
METHOD_LABELS = {
    "sda_simple_euclidean": "SDA simple, Euclidean",
    "sda_simple_weighted_euclidean": "SDA simple, weighted prox",
    "sda_weighted_euclidean": "SDA weighted, Euclidean",
    "sda_weighted_weighted_euclidean": "SDA weighted, weighted prox",
}
METHOD_COLORS = {
    "sda_simple_euclidean": "#2563eb",
    "sda_simple_weighted_euclidean": "#dc2626",
    "sda_weighted_euclidean": "#16a34a",
    "sda_weighted_weighted_euclidean": "#9333ea",
}
DASH_BY_PROFILE = {
    "none": "solid",
    "uniform": "dash",
    "coordinate_scale": "dot",
    "inverse_coordinate_scale": "dashdot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Plotly diagnostics for prox_geometry_comparison results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing iterations.csv and runs.csv. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory where HTML plots are written. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--objective",
        action="append",
        default=[],
        help="Objective id to plot. Repeat to include multiple objectives. Default: all objectives.",
    )
    parser.add_argument(
        "--prox-fun",
        action="append",
        default=[],
        help="Prox function filter, e.g. euclidean or weighted_euclidean. Default: all.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Prox weights profile filter. Repeat to include multiple. Default: all.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(TRAJECTORY_METRICS),
        default="objective_gap_x_hat",
        help="Primary metric for trajectory plots. Default: objective_gap_x_hat.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=sorted(PLOT_CHOICES),
        default=["all"],
        help="Plot groups to write. Default: all.",
    )
    parser.add_argument(
        "--max-traces-per-objective",
        type=int,
        default=500,
        help="Skip trajectory figures that would exceed this trace count. Default: 500.",
    )
    parser.add_argument(
        "--max-points-per-run",
        type=int,
        default=300,
        help="Maximum points from each run in trajectory plots. Use 0 to keep every iteration.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open generated HTML files in the default browser.",
    )
    parser.add_argument(
        "--show-traces",
        action="store_true",
        help="Show trajectory traces by default. Without this, traces start hidden in the legend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_plots = expand_plot_choices(args.plots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    iterations, runs = load_results(args.results_dir)
    runs = add_plot_metadata(runs)
    runs = filter_runs(
        runs,
        objectives=args.objective,
        prox_funs=args.prox_fun,
        profiles=args.profile,
    )
    if runs.empty:
        raise ValueError("No runs remain after filtering.")

    iterations = iterations[iterations["run_id"].isin(runs["run_id"])].copy()
    iteration_data = iterations.merge(
        runs[run_metadata_columns()],
        on="run_id",
        how="inner",
        validate="many_to_one",
    )

    written: list[Path] = []
    if "trajectories" in selected_plots:
        for objective_id in sorted(runs["objective_id"].dropna().unique()):
            objective_iterations = iteration_data[
                iteration_data["objective_id"] == objective_id
            ].copy()
            if objective_iterations.empty:
                continue
            written.append(
                write_trajectory_figure(
                    objective_id,
                    objective_iterations,
                    args.metric,
                    args.out_dir,
                    args.max_traces_per_objective,
                    args.max_points_per_run,
                    visible_by_default=args.show_traces,
                )
            )

    if "runtime-gap" in selected_plots:
        written.append(write_runtime_objective_gap_scatter(runs, args.out_dir))

    print(f"Wrote {len(written)} Plotly HTML file(s) to {args.out_dir}")
    for path in written:
        print(path)
    if args.show:
        for path in written:
            webbrowser.open(path.resolve().as_uri())


def expand_plot_choices(values: Iterable[str]) -> set[str]:
    requested = set(values)
    if "all" in requested:
        return {"trajectories", "runtime-gap"}
    return requested


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    iterations_path = results_dir / "iterations.csv"
    runs_path = results_dir / "runs.csv"
    for path in (iterations_path, runs_path):
        if not path.exists():
            raise FileNotFoundError(path)

    iterations = pd.read_csv(iterations_path)
    runs = pd.read_csv(runs_path)
    require_columns(
        iterations,
        {
            "run_id",
            "iteration",
            "objective_value_x",
            "objective_value_x_hat",
            "objective_gap_x",
            "objective_gap_x_hat",
            "normalized_gap",
        },
        iterations_path,
    )
    require_columns(
        runs,
        {
            "run_id",
            "method",
            "objective_id",
            "prox_fun",
            "prox_weights_profile",
            "dual_averaging",
            "D",
            "gamma_mult",
            "restrict_to_fd",
            "iterations",
            "converged",
            "runtime_seconds",
            "objective_gap",
            "final_gap",
        },
        runs_path,
    )
    return iterations, runs


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def add_plot_metadata(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["method_label"] = runs["method"].map(METHOD_LABELS).fillna(runs["method"])
    runs["profile_label"] = runs["prox_weights_profile"].fillna("none").replace("", "none")
    runs["D_label"] = runs["D"].map(lambda value: f"D={format_numeric(value)}")
    runs["fd_label"] = runs["restrict_to_fd"].map(
        lambda value: "FD restricted" if bool(value) else "unrestricted"
    )
    runs["tuning_label"] = runs["gamma_mult"].map(
        lambda value: f"gamma_mult={format_numeric(value)}"
    )
    runs["config_label"] = runs.apply(
        lambda row: (
            f"{row['method_label']} | profile={row['profile_label']} | "
            f"{row['tuning_label']} | {row['D_label']} | {row['fd_label']}"
        ),
        axis=1,
    )
    return runs


def filter_runs(
    runs: pd.DataFrame,
    *,
    objectives: list[str],
    prox_funs: list[str],
    profiles: list[str],
) -> pd.DataFrame:
    filtered = runs.copy()
    if objectives:
        unknown = set(objectives) - set(filtered["objective_id"].dropna().unique())
        if unknown:
            raise ValueError(f"Unknown objective id(s): {', '.join(sorted(unknown))}")
        filtered = filtered[filtered["objective_id"].isin(objectives)].copy()
    if prox_funs:
        unknown = set(prox_funs) - set(filtered["prox_fun"].dropna().unique())
        if unknown:
            raise ValueError(f"Unknown prox function(s): {', '.join(sorted(unknown))}")
        filtered = filtered[filtered["prox_fun"].isin(prox_funs)].copy()
    if profiles:
        unknown = set(profiles) - set(filtered["profile_label"].dropna().unique())
        if unknown:
            raise ValueError(f"Unknown profile(s): {', '.join(sorted(unknown))}")
        filtered = filtered[filtered["profile_label"].isin(profiles)].copy()
    return filtered


def run_metadata_columns() -> list[str]:
    return [
        "run_id",
        "method",
        "method_label",
        "objective_id",
        "prox_fun",
        "prox_weights_profile",
        "profile_label",
        "dual_averaging",
        "D",
        "D_label",
        "restrict_to_fd",
        "fd_label",
        "gamma_mult",
        "tuning_label",
        "config_label",
        "iterations",
        "converged",
        "runtime_seconds",
        "objective_gap",
        "final_gap",
    ]


def write_trajectory_figure(
    objective_id: str,
    data: pd.DataFrame,
    metric: str,
    out_dir: Path,
    max_traces: int,
    max_points_per_run: int,
    *,
    visible_by_default: bool,
) -> Path:
    metrics = [metric]
    if metric == "objective_gap_x_hat":
        metrics.append("objective_gap_x")
    elif metric == "objective_gap_x":
        metrics.append("objective_gap_x_hat")
    metrics = [column for column in dict.fromkeys(metrics) if column in data]
    fig = trajectory_figure(
        data,
        metrics=metrics,
        title=f"{objective_id}: objective trajectories by prox geometry",
        y_axis_title=metric_axis_title(metrics[0]),
        max_traces=max_traces,
        max_points_per_run=max_points_per_run,
        visible_by_default=visible_by_default,
    )
    if any(metric.startswith("objective_gap") for metric in metrics):
        fig.add_hline(y=0.0, line_dash="dash", line_color="#111827", opacity=0.55)
    path = out_dir / f"trajectories_{slug(objective_id)}.html"
    write_figure(fig, path)
    return path


def trajectory_figure(
    data: pd.DataFrame,
    *,
    metrics: list[str],
    title: str,
    y_axis_title: str,
    max_traces: int,
    max_points_per_run: int,
    visible_by_default: bool,
) -> go.Figure:
    trace_count = int(data["run_id"].nunique()) * len(metrics)
    if trace_count > max_traces:
        raise ValueError(
            f"{title} would create {trace_count} traces, exceeding --max-traces-per-objective={max_traces}."
        )

    fig = go.Figure()
    metric_dash = metric_dash_lookup(metrics)
    for method in methods_present(data):
        method_data = data[data["method"] == method]
        for run_id, run_data in method_data.groupby("run_id", sort=False):
            run_data = run_data.sort_values("iteration")
            run_data = downsample_run(run_data, max_points_per_run)
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
                            first["profile_label"],
                            first["D_label"],
                            first["fd_label"],
                            first["tuning_label"],
                            bool(first["converged"]),
                            float(first["runtime_seconds"]),
                            float(first["objective_gap"]),
                            float(first["final_gap"]),
                            metric,
                        ],
                        hovertemplate=(
                            "%{meta[0]}<br>"
                            "profile=%{meta[1]}<br>"
                            "%{meta[2]}<br>"
                            "%{meta[3]}<br>"
                            "%{meta[4]}<br>"
                            "metric=%{meta[9]}<br>"
                            "iteration=%{x}<br>"
                            "value=%{y:.6g}<br>"
                            "converged=%{meta[5]}<br>"
                            "runtime=%{meta[6]:.6g}s<br>"
                            "objective_gap=%{meta[7]:.6g}<br>"
                            "certificate_gap=%{meta[8]:.6g}"
                            "<extra></extra>"
                        ),
                    )
                )

    use_log_y = can_use_log_scale(data, metrics)
    fig.update_xaxes(
        title_text="Iteration",
        range=axis_range(data["iteration"], log_scale=False),
    )
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
        margin={"l": 80, "r": 460, "t": 80, "b": 60},
    )
    return fig


def write_runtime_objective_gap_scatter(runs: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.scatter(
        runs,
        x="runtime_seconds",
        y="objective_gap",
        color="method_label",
        symbol="profile_label",
        facet_col="objective_id",
        hover_data=[
            "prox_fun",
            "prox_weights_profile",
            "dual_averaging",
            "D",
            "gamma_mult",
            "restrict_to_fd",
            "iterations",
            "converged",
            "final_gap",
        ],
        title="Runtime vs final objective gap by prox geometry",
        labels={
            "runtime_seconds": "Runtime (seconds)",
            "objective_gap": "Final objective gap",
            "method_label": "Method",
            "profile_label": "Prox weights profile",
            "objective_id": "Objective",
            "final_gap": "SDA certificate gap",
        },
    )
    if (runs["objective_gap"].dropna() > 0).all():
        fig.update_yaxes(type="log")
    fig.update_layout(template="plotly_white", height=650)
    path = out_dir / "aggregate_runtime_vs_final_objective_gap.html"
    write_figure(fig, path)
    return path


def methods_present(data: pd.DataFrame) -> list[str]:
    present = set(data["method"].dropna().unique())
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def metric_dash_lookup(metrics: list[str]) -> dict[str, str]:
    dash_sequence = ["solid", "dash", "dot", "dashdot"]
    return {
        metric: dash_sequence[index % len(dash_sequence)]
        for index, metric in enumerate(metrics)
    }


def dash_for_run(row: pd.Series, metric_dash: str) -> str:
    if metric_dash != "solid":
        return metric_dash
    return DASH_BY_PROFILE.get(str(row["profile_label"]), "solid")


def metric_axis_title(metric: str) -> str:
    labels = {
        "objective_gap_x_hat": "Objective gap at x_hat",
        "objective_gap_x": "Objective gap at x",
        "objective_value_x_hat": "Objective value at x_hat",
        "objective_value_x": "Objective value at x",
        "normalized_gap": "Normalized SDA certificate gap",
    }
    return labels.get(metric, metric.replace("_", " "))


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


def format_numeric(value: object) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def write_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


if __name__ == "__main__":
    main()
