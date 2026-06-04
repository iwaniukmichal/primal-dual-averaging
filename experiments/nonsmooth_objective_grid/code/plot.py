from __future__ import annotations

import argparse
import math
import sys
import webbrowser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _shared.bootstrap import PROJECT_ROOT  # noqa: F401
from _shared.objectives import build_registry_objective


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_OUT_DIR = EXPERIMENT_DIR / "plots"

TRAJECTORY_METRICS = {
    "objective_value_x",
    "objective_value_x_hat",
    "objective_gap_x",
    "objective_gap_x_hat",
    "normalized_gap",
    "x_norm",
    "x_hat_norm",
    "g_norm",
}
PLOT_CHOICES = {"gaps", "norms", "gradients", "aggregates", "all"}
METHOD_ORDER = [
    "sda_simple_euclidean",
    "sda_weighted_euclidean_dual_average",
    "projected_subgradient",
]
METHOD_LABELS = {
    "sda_simple_euclidean": "SDA simple",
    "sda_weighted_euclidean_dual_average": "SDA weighted",
    "projected_subgradient": "Projected subgradient",
}
METHOD_COLORS = {
    "sda_simple_euclidean": "#2563eb",
    "sda_weighted_euclidean_dual_average": "#dc2626",
    "projected_subgradient": "#16a34a",
}
COLOR_SEQUENCE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
]
DASH_BY_D = {
    0.5: "solid",
    2.0: "dash",
    8.0: "dot",
    32.0: "dashdot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Plotly diagnostics for nonsmooth_objective_grid results."
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
        "--objective",
        action="append",
        default=[],
        help="Objective id to plot. Repeat to include multiple objectives. Default: all objectives.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(TRAJECTORY_METRICS),
        default="objective_gap_x_hat",
        help="Primary metric for gap trajectory plots. Default: objective_gap_x_hat.",
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
        help=(
            "Maximum points from each run in trajectory plots. "
            "Use 0 to keep every iteration. Default: 300."
        ),
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

    iterations, runs, summary = load_results(args.results_dir)
    if args.objective:
        objective_filter = set(args.objective)
        runs = runs[runs["objective_id"].isin(objective_filter)].copy()
        iterations = iterations[iterations["run_id"].isin(runs["run_id"])].copy()
        summary = summary[summary["objective_id"].isin(objective_filter)].copy()
        missing = sorted(objective_filter - set(runs["objective_id"].dropna().unique()))
        if missing:
            raise ValueError(f"Unknown objective id(s): {', '.join(missing)}")

    if runs.empty:
        raise ValueError("No runs remain after filtering.")

    runs = add_plot_metadata(runs)
    iteration_data = iterations.merge(
        runs[run_metadata_columns()],
        on="run_id",
        how="inner",
        validate="many_to_one",
    )

    written: list[Path] = []
    for objective_id in sorted(runs["objective_id"].dropna().unique()):
        objective_iterations = iteration_data[
            iteration_data["objective_id"] == objective_id
        ].copy()
        objective_runs = runs[runs["objective_id"] == objective_id].copy()
        if objective_iterations.empty:
            continue

        if "gaps" in selected_plots:
            written.append(
                write_gap_figure(
                    objective_id,
                    objective_iterations,
                    args.metric,
                    args.out_dir,
                    args.max_traces_per_objective,
                    args.max_points_per_run,
                    visible_by_default=args.show_traces,
                )
            )
        if "norms" in selected_plots:
            written.append(
                write_norm_figure(
                    objective_id,
                    objective_iterations,
                    args.out_dir,
                    args.max_traces_per_objective,
                    args.max_points_per_run,
                    visible_by_default=args.show_traces,
                )
            )
        if "gradients" in selected_plots:
            written.append(
                write_gradient_figure(
                    objective_id,
                    objective_iterations,
                    args.out_dir,
                    args.max_traces_per_objective,
                    args.max_points_per_run,
                    visible_by_default=args.show_traces,
                )
            )

        # Keep the per-objective run frame available for future extensions and
        # validate that every trajectory objective has run metadata.
        if objective_runs.empty:
            raise ValueError(f"No run metadata found for {objective_id}.")

    if "aggregates" in selected_plots:
        written.extend(write_aggregate_figures(runs, summary, args.out_dir))

    print(f"Wrote {len(written)} Plotly HTML file(s) to {args.out_dir}")
    for path in written:
        print(path)
    if args.show:
        for path in written:
            webbrowser.open(path.resolve().as_uri())


def expand_plot_choices(values: Iterable[str]) -> set[str]:
    requested = set(values)
    if "all" in requested:
        return {"gaps", "norms", "gradients", "aggregates"}
    return requested


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    iterations_path = results_dir / "iterations.csv"
    runs_path = results_dir / "runs.csv"
    summary_path = results_dir / "summary.csv"
    for path in (iterations_path, runs_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    iterations = pd.read_csv(iterations_path)
    runs = pd.read_csv(runs_path)
    summary = pd.read_csv(summary_path)

    require_columns(
        iterations,
        {
            "run_id",
            "iteration",
            "objective_gap_x",
            "objective_gap_x_hat",
            "x_norm",
            "x_hat_norm",
            "g_norm",
        },
        iterations_path,
    )
    require_columns(
        runs,
        {
            "run_id",
            "method",
            "objective_id",
            "D",
            "gamma_mult",
            "alpha",
            "restrict_to_fd",
            "iterations",
            "converged",
            "runtime_seconds",
            "objective_gap",
            "final_gap",
        },
        runs_path,
    )
    require_columns(
        summary,
        {
            "method",
            "objective_id",
            "mean_runtime_seconds",
            "mean_final_gap",
        },
        summary_path,
    )
    return iterations, runs, summary


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def add_plot_metadata(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["method_label"] = runs["method"].map(METHOD_LABELS).fillna(runs["method"])
    runs["tuning_name"] = np.where(runs["alpha"].notna(), "alpha", "gamma_mult")
    runs["tuning_value"] = runs["alpha"].where(runs["alpha"].notna(), runs["gamma_mult"])
    runs["tuning_label"] = runs.apply(
        lambda row: f"{row['tuning_name']}={format_numeric(row['tuning_value'])}",
        axis=1,
    )
    runs["D_label"] = runs["D"].map(lambda value: f"D={format_numeric(value)}")
    runs["fd_label"] = runs["restrict_to_fd"].map(
        lambda value: "FD restricted" if bool(value) else "unrestricted"
    )
    runs["config_label"] = runs.apply(
        lambda row: (
            f"{row['method_label']} | {row['tuning_label']} | "
            f"{row['D_label']} | {row['fd_label']}"
        ),
        axis=1,
    )
    return runs


def run_metadata_columns() -> list[str]:
    return [
        "run_id",
        "method",
        "method_label",
        "objective_id",
        "D",
        "D_label",
        "restrict_to_fd",
        "fd_label",
        "gamma_mult",
        "alpha",
        "tuning_name",
        "tuning_value",
        "tuning_label",
        "config_label",
        "iterations",
        "converged",
        "runtime_seconds",
        "objective_gap",
        "final_gap",
    ]


def write_gap_figure(
    objective_id: str,
    data: pd.DataFrame,
    metric: str,
    out_dir: Path,
    max_traces: int,
    max_points_per_run: int,
    *,
    visible_by_default: bool,
) -> Path:
    metric_columns = [metric]
    if metric == "objective_gap_x_hat":
        metric_columns.append("objective_gap_x")
    elif metric == "objective_gap_x":
        metric_columns.append("objective_gap_x_hat")
    metric_columns = [column for column in dict.fromkeys(metric_columns) if column in data]

    fig = trajectory_subplots(
        data,
        metrics=metric_columns,
        title=f"{objective_id}: objective trajectories",
        y_axis_title="Objective gap" if "gap" in metric else metric.replace("_", " "),
        max_traces=max_traces,
        max_points_per_run=max_points_per_run,
        visible_by_default=visible_by_default,
    )
    path = out_dir / f"trajectories_{slug(objective_id)}.html"
    write_figure(fig, path)
    return path


def write_norm_figure(
    objective_id: str,
    data: pd.DataFrame,
    out_dir: Path,
    max_traces: int,
    max_points_per_run: int,
    *,
    visible_by_default: bool,
) -> Path:
    fig = trajectory_subplots(
        data,
        metrics=["x_norm", "x_hat_norm"],
        title=f"{objective_id}: iterate norm trajectories",
        y_axis_title="Euclidean norm",
        max_traces=max_traces,
        max_points_per_run=max_points_per_run,
        visible_by_default=visible_by_default,
        log_y=False,
    )
    optimum_norm = value_norm(build_registry_objective(objective_id).minimizer)
    fig.add_hline(
        y=optimum_norm,
        line_dash="dash",
        line_color="#111827",
        opacity=0.65,
        annotation_text=f"||x*||={format_numeric(optimum_norm)}",
        annotation_position="top left",
    )
    fig.update_yaxes(
        range=axis_range(
            pd.concat([data[["x_norm", "x_hat_norm"]].stack().dropna(), pd.Series([optimum_norm])]),
            log_scale=False,
        )
    )
    path = out_dir / f"norms_{slug(objective_id)}.html"
    write_figure(fig, path)
    return path


def write_gradient_figure(
    objective_id: str,
    data: pd.DataFrame,
    out_dir: Path,
    max_traces: int,
    max_points_per_run: int,
    *,
    visible_by_default: bool,
) -> Path:
    fig = trajectory_subplots(
        data,
        metrics=["g_norm"],
        title=f"{objective_id}: subgradient norm trajectories",
        y_axis_title="Subgradient norm",
        max_traces=max_traces,
        max_points_per_run=max_points_per_run,
        visible_by_default=visible_by_default,
    )
    path = out_dir / f"gradients_{slug(objective_id)}.html"
    write_figure(fig, path)
    return path


def trajectory_subplots(
    data: pd.DataFrame,
    *,
    metrics: list[str],
    title: str,
    y_axis_title: str,
    max_traces: int,
    max_points_per_run: int,
    visible_by_default: bool,
    log_y: bool | None = None,
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
                legend_name = f"{first['config_label']} | {metric}"
                legend_key = f"{first['run_id']}::{metric}"
                fig.add_trace(
                    go.Scattergl(
                        x=series["iteration"],
                        y=series[metric],
                        mode="lines+markers",
                        name=legend_name,
                        legendgroup=legend_key,
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
                        opacity=0.72 if bool(first["restrict_to_fd"]) else 0.95,
                        meta=[
                            first["method_label"],
                            first["D_label"],
                            first["fd_label"],
                            first["tuning_label"],
                            bool(first["converged"]),
                            float(first["runtime_seconds"]),
                            float(first["final_gap"]),
                            metric,
                        ],
                        hovertemplate=(
                            "%{meta[0]}<br>"
                            "%{meta[1]}<br>"
                            "%{meta[2]}<br>"
                            "%{meta[3]}<br>"
                            "metric=%{meta[7]}<br>"
                            "iteration=%{x}<br>"
                            "value=%{y:.6g}<br>"
                            "converged=%{meta[4]}<br>"
                            "runtime=%{meta[5]:.6g}s<br>"
                            "final_gap=%{meta[6]:.6g}"
                            "<extra></extra>"
                        ),
                    ),
                )

    use_log_y = can_use_log_scale(data, metrics) if log_y is None else log_y
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
        margin={"l": 80, "r": 360, "t": 80, "b": 60},
    )
    return fig


def methods_present(data: pd.DataFrame) -> list[str]:
    present = set(data["method"].dropna().unique())
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def color_lookup(data: pd.DataFrame) -> dict[str, str]:
    labels = sorted(
        data["tuning_label"].dropna().unique(),
        key=lambda label: (
            label.split("=")[0],
            float(label.split("=")[1]) if "=" in label else math.inf,
        ),
    )
    return {
        str(label): COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)]
        for index, label in enumerate(labels)
    }


def metric_dash_lookup(metrics: list[str]) -> dict[str, str]:
    dash_sequence = ["solid", "dash", "dot", "dashdot"]
    return {
        metric: dash_sequence[index % len(dash_sequence)]
        for index, metric in enumerate(metrics)
    }


def dash_for_run(row: pd.Series, metric_dash: str) -> str:
    d_dash = DASH_BY_D.get(float(row["D"]), "solid")
    if bool(row["restrict_to_fd"]):
        if d_dash == "solid":
            return "longdash"
        if d_dash == "dash":
            return "longdashdot"
    if metric_dash != "solid":
        return metric_dash
    return d_dash


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




def write_aggregate_figures(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    paths.append(write_final_gap_heatmaps(runs, out_dir))
    paths.append(write_runtime_gap_scatter(runs, out_dir))
    paths.append(write_runtime_objective_gap_scatter(runs, out_dir))
    paths.append(write_convergence_bars(runs, out_dir))
    paths.append(write_summary_comparison(summary, out_dir))
    return paths


def write_final_gap_heatmaps(runs: pd.DataFrame, out_dir: Path) -> Path:
    heatmap_data = runs.copy()
    heatmap_data["step_parameter"] = heatmap_data["tuning_label"]
    aggregated = (
        heatmap_data.groupby(
            [
                "objective_id",
                "method_label",
                "restrict_to_fd",
                "D",
                "step_parameter",
            ],
            dropna=False,
            as_index=False,
        )["final_gap"]
        .mean()
        .sort_values(["objective_id", "method_label", "restrict_to_fd", "D"])
    )
    fig = px.density_heatmap(
        aggregated,
        x="step_parameter",
        y="D",
        z="final_gap",
        facet_row="method_label",
        facet_col="restrict_to_fd",
        animation_frame="objective_id",
        histfunc="avg",
        color_continuous_scale="Viridis",
        title="Mean final gap by D and step parameter",
        labels={
            "step_parameter": "Step parameter",
            "D": "D",
            "final_gap": "Mean final gap",
            "restrict_to_fd": "FD restricted",
            "method_label": "Method",
        },
    )
    fig.update_layout(template="plotly_white", height=900)
    path = out_dir / "aggregate_final_gap_heatmaps.html"
    write_figure(fig, path)
    return path


def write_runtime_gap_scatter(runs: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.scatter(
        runs,
        x="runtime_seconds",
        y="final_gap",
        color="method_label",
        symbol="converged",
        facet_col="objective_id",
        facet_col_wrap=3,
        hover_data=[
            "D",
            "restrict_to_fd",
            "gamma_mult",
            "alpha",
            "iterations",
            "tuning_label",
            "objective_gap",
        ],
        title="Runtime vs final gap by run",
        labels={
            "runtime_seconds": "Runtime (seconds)",
            "final_gap": "Final gap",
            "method_label": "Method",
            "converged": "Converged",
            "objective_gap": "Final objective gap",
        },
    )
    if (runs["final_gap"].dropna() > 0).all():
        fig.update_yaxes(type="log")
    fig.update_layout(template="plotly_white", height=900)
    path = out_dir / "aggregate_runtime_vs_final_gap.html"
    write_figure(fig, path)
    return path


def write_runtime_objective_gap_scatter(runs: pd.DataFrame, out_dir: Path) -> Path:
    fig = px.scatter(
        runs,
        x="runtime_seconds",
        y="objective_gap",
        color="method_label",
        facet_col="objective_id",
        facet_col_wrap=3,
        hover_data=[
            "D",
            "restrict_to_fd",
            "gamma_mult",
            "alpha",
            "iterations",
            "tuning_label",
            "converged",
            "final_gap",
        ],
        title="Runtime vs final objective gap by run",
        labels={
            "runtime_seconds": "Runtime (seconds)",
            "objective_gap": "Final objective gap",
            "method_label": "Method",
            "final_gap": "SDA certificate gap",
        },
    )
    if (runs["objective_gap"].dropna() > 0).all():
        fig.update_yaxes(type="log")
    fig.update_layout(template="plotly_white", height=900)
    path = out_dir / "aggregate_runtime_vs_final_objective_gap.html"
    write_figure(fig, path)
    return path


def write_convergence_bars(runs: pd.DataFrame, out_dir: Path) -> Path:
    convergence = (
        runs.groupby(["objective_id", "method_label"], as_index=False)["converged"]
        .mean()
        .assign(converged_percent=lambda frame: 100.0 * frame["converged"])
    )
    fig = px.bar(
        convergence,
        x="objective_id",
        y="converged_percent",
        color="method_label",
        barmode="group",
        title="Convergence rate by objective and method",
        labels={
            "objective_id": "Objective",
            "converged_percent": "Converged runs (%)",
            "method_label": "Method",
        },
    )
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(template="plotly_white", height=650)
    path = out_dir / "aggregate_convergence_rates.html"
    write_figure(fig, path)
    return path


def write_summary_comparison(summary: pd.DataFrame, out_dir: Path) -> Path:
    summary = summary.copy()
    summary["method_label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=["Mean final gap", "Mean runtime"],
    )
    for method in methods_present(summary.rename(columns={"method": "method"})):
        method_data = summary[summary["method"] == method].sort_values("objective_id")
        label = METHOD_LABELS.get(method, method)
        fig.add_trace(
            go.Bar(
                x=method_data["objective_id"],
                y=method_data["mean_final_gap"],
                name=label,
                legendgroup=label,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=method_data["objective_id"],
                y=method_data["mean_runtime_seconds"],
                name=label,
                legendgroup=label,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="Mean final gap", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Mean runtime (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Objective", row=2, col=1)
    fig.update_layout(
        title="Summary metrics by objective and method",
        barmode="group",
        template="plotly_white",
        height=850,
    )
    path = out_dir / "aggregate_summary_comparison.html"
    write_figure(fig, path)
    return path


def write_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def value_norm(value: object) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.linalg.norm(array))


def format_numeric(value: object) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


if __name__ == "__main__":
    main()
