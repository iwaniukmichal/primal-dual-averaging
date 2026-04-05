from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "plot"


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _to_float_array(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _series_axis(length: int) -> list[float]:
    return list(np.arange(1, length + 1, dtype=float))


def _time_axis(run: dict[str, Any], length: int) -> list[float] | None:
    avg_time = run.get("avg_iteration_time_seconds")
    if avg_time is None:
        return None
    return list(np.arange(1, length + 1, dtype=float) * float(avg_time))


def _iterates_for_plot(run: dict[str, Any]) -> tuple[list[float], np.ndarray, np.ndarray]:
    iterations = int(run["iterations"])
    x = _to_float_array(run["x"][:iterations])
    x_hat = _to_float_array(run["x_hat"][1 : iterations + 1])
    return _series_axis(iterations), x, x_hat


def _reference_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _detect_schema(runs: list[dict[str, Any]]) -> str:
    if any(run.get("method") == "sklearn" or "train_loss_x" in run for run in runs):
        return "logreg"
    if runs and all("objective_minimum_value" in run for run in runs):
        return "bench"
    return "unknown"


def _objective_identity(schema: str, run: dict[str, Any]) -> dict[str, Any]:
    if schema == "bench":
        return {
            "objective_id": run.get("objective_id"),
            "objective_family": run.get("objective_family"),
            "objective_params": run.get("objective_params"),
            "objective_dimension": run.get("objective_dimension"),
        }
    return {
        "objective_id": run.get("objective_id"),
        "objective_family": run.get("objective_family"),
        "objective_params": run.get("objective_params"),
        "objective_dimension": run.get("objective_dimension"),
        "dataset_path": run.get("objective_params", {}).get("dataset_path"),
    }


def _partition_runs(schema: str, runs: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        partitions[_stable_json(_objective_identity(schema, run))].append(run)
    return sorted(partitions.items(), key=lambda item: item[0])


def _fallback_run_key(run: dict[str, Any]) -> str:
    return _stable_json(
        {
            "method": run.get("method"),
            "solver_name": run.get("solver_name"),
            "objective_id": run.get("objective_id"),
            "objective_params": run.get("objective_params"),
            "D": run.get("D"),
            "gamma_multiplier": run.get("gamma_multiplier"),
            "alpha": run.get("alpha"),
            "restrict_to_fd": run.get("restrict_to_fd"),
        }
    )


def _deduplicate_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for run in runs:
        key = (str(run.get("run_id") or _fallback_run_key(run)), str(run["method"]))
        if key in seen:
            duplicates.append(
                {
                    "method": run["method"],
                    "run_id": run.get("run_id"),
                    "fallback_key": _fallback_run_key(run) if run.get("run_id") is None else None,
                }
            )
            continue
        seen.add(key)
        deduped.append(run)
    return deduped, duplicates


def _partition_plot_prefix(partition_run: dict[str, Any], *, total_partitions: int) -> str:
    if total_partitions == 1:
        return ""
    return f"{_slugify(str(partition_run.get('objective_id', 'partition')))}__"


def _run_label(run: dict[str, Any]) -> str:
    run_id = run.get("run_id")
    if run_id:
        run_id_text = str(run_id)
    else:
        if run.get("method") == "sda":
            parts = [
                f"D={_format_number(run.get('D'))}",
                f"gamma={_format_number(run.get('gamma_multiplier'))}",
            ]
        elif run.get("method") == "subgradient":
            parts = [
                f"D={_format_number(run.get('D'))}",
                f"alpha={_format_number(run.get('alpha'))}",
            ]
        else:
            parts = [str(run.get("solver_name", run.get("method", "run")))]
        if run.get("restrict_to_fd") is not None:
            parts.append(f"restrict_to_fd={run.get('restrict_to_fd')}")
        run_id_text = "legacy:" + "|".join(parts)
    method = str(run["method"])
    if method == "sda":
        details = f"D={_format_number(run['D'])}, gamma={_format_number(run['gamma_multiplier'])}"
    elif method == "subgradient":
        details = f"D={_format_number(run['D'])}, alpha={_format_number(run['alpha'])}"
    else:
        solver = str(run.get("solver_name", "sklearn"))
        details = solver
    return f"{run_id_text} | {method} | {details}"


def _plotly_dash(style: str) -> str:
    if style == "dashed":
        return "dash"
    if style == "dotted":
        return "dot"
    return "solid"


def _mpl_linestyle(style: str) -> str:
    if style == "dashed":
        return "--"
    if style == "dotted":
        return ":"
    return "-"


def _ensure_matplotlib(plot_root: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    cache_root = plot_root / ".plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _make_series(
    x_axis: Sequence[float],
    y_axis: Sequence[float],
    *,
    label: str,
    style: str = "solid",
    source_label: str | None = None,
) -> dict[str, Any]:
    return {
        "x": list(x_axis),
        "y": [float(value) for value in y_axis],
        "label": label,
        "style": style,
        "source_label": source_label,
    }


def _make_constant_series(
    x_axis: Sequence[float],
    value: float,
    *,
    label: str,
) -> dict[str, Any]:
    return {
        "x": list(x_axis),
        "y": [float(value)] * len(x_axis),
        "label": label,
        "style": "dashed",
        "source_label": None,
    }


def _record_generated_plot(
    group_manifest: dict[str, Any],
    *,
    plot_id: str,
    title: str,
    png_path: Path,
    html_path: Path,
    series_labels: list[str],
) -> None:
    group_manifest["generated_plots"].append(
        {
            "plot_id": plot_id,
            "title": title,
            "png": png_path.name,
            "html": html_path.name,
            "series_labels": series_labels,
        }
    )


def _record_skipped_plot(group_manifest: dict[str, Any], *, plot_id: str, reason: str) -> None:
    group_manifest["skipped_plots"].append({"plot_id": plot_id, "reason": reason})


def _write_line_plot(
    *,
    plot_root: Path,
    group_manifest: dict[str, Any],
    plot_id: str,
    title: str,
    x_label: str,
    y_label: str,
    panel_titles: list[str],
    panels: list[list[dict[str, Any]]],
) -> None:
    png_dir = plot_root / "png"
    html_dir = plot_root / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    series_labels: list[str] = []
    for panel in panels:
        for series in panel:
            source_label = series.get("source_label")
            if source_label and source_label not in series_labels:
                series_labels.append(source_label)

    if not any(panel for panel in panels):
        _record_skipped_plot(
            group_manifest,
            plot_id=plot_id,
            reason="no comparable runs available for this plot",
        )
        return

    plt = _ensure_matplotlib(plot_root)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13, max(4.5, 3.5 * len(panels))),
        squeeze=False,
    )
    for index, ax in enumerate(axes[:, 0]):
        if panel_titles[index]:
            ax.set_title(panel_titles[index])
        for series in panels[index]:
            ax.plot(
                series["x"],
                series["y"],
                label=series["label"],
                linestyle=_mpl_linestyle(series.get("style", "solid")),
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if panels[index]:
            ax.legend(loc="best", fontsize="x-small")
    fig.suptitle(title)
    fig.tight_layout()

    png_path = png_dir / f"{plot_id}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = make_subplots(
        rows=len(panels),
        cols=1,
        subplot_titles=panel_titles,
        shared_xaxes=False,
    )
    for row_index, panel in enumerate(panels, start=1):
        for series in panel:
            plotly_fig.add_trace(
                go.Scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines",
                    name=series["label"],
                    line={"dash": _plotly_dash(series.get("style", "solid"))},
                    showlegend=row_index == 1,
                ),
                row=row_index,
                col=1,
            )
        plotly_fig.update_xaxes(title_text=x_label, row=row_index, col=1)
        plotly_fig.update_yaxes(title_text=y_label, row=row_index, col=1)
    plotly_fig.update_layout(title=title, height=max(450, 320 * len(panels)))
    html_path = html_dir / f"{plot_id}.html"
    plotly_fig.write_html(str(html_path), include_plotlyjs="cdn")

    _record_generated_plot(
        group_manifest,
        plot_id=plot_id,
        title=title,
        png_path=png_path,
        html_path=html_path,
        series_labels=series_labels,
    )


def _write_bar_plot(
    *,
    plot_root: Path,
    group_manifest: dict[str, Any],
    plot_id: str,
    title: str,
    x_label: str,
    y_label: str,
    labels: list[str],
    values: list[float],
) -> None:
    if not labels:
        _record_skipped_plot(
            group_manifest,
            plot_id=plot_id,
            reason="no comparable runs available for this plot",
        )
        return

    png_dir = plot_root / "png"
    html_dir = plot_root / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    plt = _ensure_matplotlib(plot_root)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    png_path = png_dir / f"{plot_id}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = go.Figure(data=[go.Bar(x=labels, y=values)])
    plotly_fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    html_path = html_dir / f"{plot_id}.html"
    plotly_fig.write_html(str(html_path), include_plotlyjs="cdn")

    _record_generated_plot(
        group_manifest,
        plot_id=plot_id,
        title=title,
        png_path=png_path,
        html_path=html_path,
        series_labels=labels,
    )


def _metric_panels_for_runs(
    runs: list[dict[str, Any]],
    *,
    primary_key: str,
    secondary_key: str | None = None,
    suffix_primary: str = "x",
    suffix_secondary: str = "x_hat",
) -> list[list[dict[str, Any]]]:
    panels = [[]]
    for run in runs:
        if primary_key not in run:
            continue
        x_axis = _series_axis(len(run[primary_key]))
        label = _run_label(run)
        panels[0].append(
            _make_series(
                x_axis,
                run[primary_key],
                label=f"{label} | {suffix_primary}",
                source_label=label,
            )
        )
        if secondary_key is not None and secondary_key in run:
            panels[0].append(
                _make_series(
                    x_axis,
                    run[secondary_key],
                    label=f"{label} | {suffix_secondary}",
                    style="dashed",
                    source_label=label,
                )
            )
    return panels


def _add_constant_line(
    panels: list[list[dict[str, Any]]],
    *,
    value: float,
    label: str,
) -> None:
    if not panels or not panels[0]:
        return
    x_axis = panels[0][0]["x"]
    panels[0].append(_make_constant_series(x_axis, value, label=label))


def _maybe_add_time_plot(
    *,
    plot_root: Path,
    group_manifest: dict[str, Any],
    plot_id: str,
    title: str,
    y_label: str,
    runs: list[dict[str, Any]],
    primary_key: str,
    secondary_key: str | None = None,
    baseline_runs: list[dict[str, Any]] | None = None,
    baseline_key: str | None = None,
) -> None:
    panels = [[]]
    any_timed_run = False
    for run in runs:
        if primary_key not in run:
            continue
        x_axis = _time_axis(run, len(run[primary_key]))
        if x_axis is None:
            continue
        any_timed_run = True
        label = _run_label(run)
        panels[0].append(
            _make_series(
                x_axis,
                run[primary_key],
                label=f"{label} | x",
                source_label=label,
            )
        )
        if secondary_key is not None and secondary_key in run:
            panels[0].append(
                _make_series(
                    x_axis,
                    run[secondary_key],
                    label=f"{label} | x_hat",
                    style="dashed",
                    source_label=label,
                )
            )
    if baseline_runs and baseline_key and panels[0]:
        max_x = max(max(series["x"]) for series in panels[0])
        baseline_axis = [0.0, float(max_x)]
        for baseline_run in baseline_runs:
            if baseline_key not in baseline_run:
                continue
            label = _run_label(baseline_run)
            panels[0].append(
                _make_constant_series(
                    baseline_axis,
                    float(baseline_run[baseline_key]),
                    label=f"{label} | baseline",
                )
            )
    if not any_timed_run:
        _record_skipped_plot(
            group_manifest,
            plot_id=plot_id,
            reason="timing fields missing from saved runs",
        )
        return
    _write_line_plot(
        plot_root=plot_root,
        group_manifest=group_manifest,
        plot_id=plot_id,
        title=title,
        x_label="Cumulative time (s)",
        y_label=y_label,
        panel_titles=[""],
        panels=panels,
    )


def _bench_distance_series(run: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    x_axis, x, x_hat = _iterates_for_plot(run)
    target = _reference_vector(run["objective_minimizer"]).reshape(1, -1)
    return (
        x_axis,
        np.linalg.norm(x - target, axis=1).tolist(),
        np.linalg.norm(x_hat - target, axis=1).tolist(),
    )


def _logreg_distance_series(
    run: dict[str, Any],
    baseline_vector: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    x_axis, x, x_hat = _iterates_for_plot(run)
    target = _reference_vector(baseline_vector).reshape(1, -1)
    return (
        x_axis,
        np.linalg.norm(x - target, axis=1).tolist(),
        np.linalg.norm(x_hat - target, axis=1).tolist(),
    )


def _coordinate_indices_for_logreg(runs: list[dict[str, Any]], *, max_coordinates: int = 8) -> list[int]:
    dimension = int(runs[0]["objective_dimension"])
    if dimension <= max_coordinates:
        return list(range(dimension))

    stacked: list[np.ndarray] = []
    for run in runs:
        _, x, x_hat = _iterates_for_plot(run)
        stacked.extend([x, x_hat])
    joined = np.vstack(stacked)
    variability = np.var(joined, axis=0)
    top_variable = list(np.argsort(variability)[::-1][: max_coordinates - 1])
    bias_index = dimension - 1
    selected: list[int] = []
    for index in [*top_variable, bias_index]:
        if index not in selected:
            selected.append(int(index))
    return selected[:max_coordinates]


def _plot_bench_partition(
    *,
    plot_root: Path,
    group_manifest: dict[str, Any],
    plot_prefix: str,
    runs: list[dict[str, Any]],
) -> None:
    objective_id = str(runs[0]["objective_id"])
    objective_minimum = float(runs[0]["objective_minimum_value"])
    objective_minimizer = _reference_vector(runs[0]["objective_minimizer"])
    sda_runs = [run for run in runs if run["method"] == "sda"]
    subgradient_runs = [run for run in runs if run["method"] == "subgradient"]
    iterative_runs = [run for run in runs if run["method"] in {"sda", "subgradient"}]

    if iterative_runs:
        panels = _metric_panels_for_runs(
            iterative_runs,
            primary_key="f_x",
            secondary_key="f_x_hat",
        )
        _add_constant_line(panels, value=objective_minimum, label="f*")
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}objective_iterations",
            title=f"{objective_id} objective over iterations",
            x_label="Iteration",
            y_label="Objective value",
            panel_titles=[""],
            panels=panels,
        )

        gap_runs = []
        for run in iterative_runs:
            if run["method"] == "sda":
                gap_runs.append(
                    {
                        **run,
                        "objective_gap_x": [float(value) - objective_minimum for value in run["f_x"]],
                        "objective_gap_x_hat": [
                            float(value) - objective_minimum for value in run["f_x_hat"]
                        ],
                    }
                )
            else:
                gap_runs.append(run)
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}objective_gap_iterations",
            title=f"{objective_id} objective gap over iterations",
            x_label="Iteration",
            y_label="Objective gap",
            panel_titles=[""],
            panels=_metric_panels_for_runs(
                gap_runs,
                primary_key="objective_gap_x",
                secondary_key="objective_gap_x_hat",
            ),
        )
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}sda_normalized_gap_iterations",
            title=f"{objective_id} SDA normalized gap over iterations",
            x_label="Iteration",
            y_label="Normalized gap",
            panel_titles=[""],
            panels=_metric_panels_for_runs(
                sda_runs,
                primary_key="normalized_gap",
                secondary_key=None,
                suffix_primary="normalized_gap",
                suffix_secondary="",
            ),
        )
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}iterate_norm_iterations",
            title=f"{objective_id} iterate norms over iterations",
            x_label="Iteration",
            y_label="Norm",
            panel_titles=[""],
            panels=_metric_panels_for_runs(
                iterative_runs,
                primary_key="x_norm",
                secondary_key="x_hat_norm",
            ),
        )
        _maybe_add_time_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}objective_time",
            title=f"{objective_id} objective over time",
            y_label="Objective value",
            runs=iterative_runs,
            primary_key="f_x",
            secondary_key="f_x_hat",
            baseline_runs=[
                {
                    "run_id": "objective-optimum",
                    "method": "reference",
                    "D": None,
                    "gamma_multiplier": None,
                    "final_objective": objective_minimum,
                }
            ],
            baseline_key="final_objective",
        )
        _maybe_add_time_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}objective_gap_time",
            title=f"{objective_id} objective gap over time",
            y_label="Objective gap",
            runs=gap_runs,
            primary_key="objective_gap_x",
            secondary_key="objective_gap_x_hat",
        )

        x_panels: list[list[dict[str, Any]]] = [[] for _ in range(objective_minimizer.shape[0])]
        panel_titles = [f"Coordinate {index}" for index in range(objective_minimizer.shape[0])]
        for coord_index in range(objective_minimizer.shape[0]):
            x_axis = _series_axis(int(iterative_runs[0]["iterations"]))
            x_panels[coord_index].append(
                _make_constant_series(
                    x_axis,
                    float(objective_minimizer[coord_index]),
                    label="x*",
                )
            )
        for run in iterative_runs:
            run_label = _run_label(run)
            x_axis, x, x_hat = _iterates_for_plot(run)
            for coord_index in range(x.shape[1]):
                x_panels[coord_index].append(
                    _make_series(
                        x_axis,
                        x[:, coord_index],
                        label=f"{run_label} | x",
                        source_label=run_label,
                    )
                )
                x_panels[coord_index].append(
                    _make_series(
                        x_axis,
                        x_hat[:, coord_index],
                        label=f"{run_label} | x_hat",
                        style="dashed",
                        source_label=run_label,
                    )
                )
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}iterates_iterations",
            title=f"{objective_id} iterates over iterations",
            x_label="Iteration-aligned step",
            y_label="Coordinate value",
            panel_titles=panel_titles,
            panels=x_panels,
        )

        distance_panels = [[]]
        for run in iterative_runs:
            run_label = _run_label(run)
            dist_axis, dist_x, dist_x_hat = _bench_distance_series(run)
            distance_panels[0].append(
                _make_series(dist_axis, dist_x, label=f"{run_label} | x", source_label=run_label)
            )
            distance_panels[0].append(
                _make_series(
                    dist_axis,
                    dist_x_hat,
                    label=f"{run_label} | x_hat",
                    style="dashed",
                    source_label=run_label,
                )
            )
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}distance_iterations",
            title=f"{objective_id} distance to optimum",
            x_label="Iteration",
            y_label="Distance to x*",
            panel_titles=[""],
            panels=distance_panels,
        )

    if subgradient_runs:
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}subgradient_subgradient_norm_iterations",
            title=f"{objective_id} subgradient norm over iterations",
            x_label="Iteration",
            y_label="Subgradient norm",
            panel_titles=[""],
            panels=_metric_panels_for_runs(
                subgradient_runs,
                primary_key="g_norm",
                secondary_key=None,
                suffix_primary="g_norm",
                suffix_secondary="",
            ),
        )

    timed_runs = [run for run in runs if "total_runtime_seconds" in run]
    if timed_runs:
        _write_bar_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}runtime_summary",
            title=f"{objective_id} runtime by method",
            x_label="Run",
            y_label="Runtime (s)",
            labels=[_run_label(run) for run in timed_runs],
            values=[float(run["total_runtime_seconds"]) for run in timed_runs],
        )
    else:
        _record_skipped_plot(
            group_manifest,
            plot_id=f"{plot_prefix}runtime_summary",
            reason="timing fields missing from saved runs",
        )


def _plot_logreg_partition(
    *,
    plot_root: Path,
    group_manifest: dict[str, Any],
    plot_prefix: str,
    runs: list[dict[str, Any]],
) -> None:
    objective_id = str(runs[0]["objective_id"])
    sda_runs = [run for run in runs if run["method"] == "sda"]
    subgradient_runs = [run for run in runs if run["method"] == "subgradient"]
    sklearn_runs = [run for run in runs if run["method"] == "sklearn"]
    iterative_runs = [run for run in runs if run["method"] in {"sda", "subgradient"}]
    baseline_vector = sklearn_runs[0]["final_parameter_vector"] if sklearn_runs else None

    def add_metric_plot(
        *,
        plot_id: str,
        title: str,
        y_label: str,
        method_runs: list[dict[str, Any]],
        primary_key: str,
        secondary_key: str,
        baseline_key: str | None,
    ) -> None:
        panels = _metric_panels_for_runs(
            method_runs,
            primary_key=primary_key,
            secondary_key=secondary_key,
        )
        if baseline_key is not None and panels[0]:
            for baseline_run in sklearn_runs:
                if baseline_key not in baseline_run:
                    continue
                panels[0].append(
                    _make_constant_series(
                        panels[0][0]["x"],
                        float(baseline_run[baseline_key]),
                        label=f"{_run_label(baseline_run)} | baseline",
                    )
                )
        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}{plot_id}",
            title=title,
            x_label="Iteration",
            y_label=y_label,
            panel_titles=[""],
            panels=panels,
        )

    add_metric_plot(
        plot_id="train_loss_iterations",
        title=f"{objective_id} train loss over iterations",
        y_label="Train loss",
        method_runs=iterative_runs,
        primary_key="train_loss_x",
        secondary_key="train_loss_x_hat",
        baseline_key="final_train_loss",
    )
    add_metric_plot(
        plot_id="test_loss_iterations",
        title=f"{objective_id} test loss over iterations",
        y_label="Test loss",
        method_runs=iterative_runs,
        primary_key="test_loss_x",
        secondary_key="test_loss_x_hat",
        baseline_key="final_test_loss",
    )
    add_metric_plot(
        plot_id="test_accuracy_iterations",
        title=f"{objective_id} test accuracy over iterations",
        y_label="Test accuracy",
        method_runs=iterative_runs,
        primary_key="test_accuracy_x",
        secondary_key="test_accuracy_x_hat",
        baseline_key="final_test_accuracy",
    )
    add_metric_plot(
        plot_id="nonzero_count_iterations",
        title=f"{objective_id} nonzero count over iterations",
        y_label="Nonzero count",
        method_runs=iterative_runs,
        primary_key="nonzero_count_x",
        secondary_key="nonzero_count_x_hat",
        baseline_key="final_nonzero_count",
    )

    _write_line_plot(
        plot_root=plot_root,
        group_manifest=group_manifest,
        plot_id=f"{plot_prefix}sda_normalized_gap_iterations",
        title=f"{objective_id} SDA normalized gap over iterations",
        x_label="Iteration",
        y_label="Normalized gap",
        panel_titles=[""],
        panels=_metric_panels_for_runs(
            sda_runs,
            primary_key="normalized_gap",
            secondary_key=None,
            suffix_primary="normalized_gap",
            suffix_secondary="",
        ),
    )

    _maybe_add_time_plot(
        plot_root=plot_root,
        group_manifest=group_manifest,
        plot_id=f"{plot_prefix}train_loss_time",
        title=f"{objective_id} train loss over time",
        y_label="Train loss",
        runs=iterative_runs,
        primary_key="train_loss_x",
        secondary_key="train_loss_x_hat",
        baseline_runs=sklearn_runs,
        baseline_key="final_train_loss",
    )
    _maybe_add_time_plot(
        plot_root=plot_root,
        group_manifest=group_manifest,
        plot_id=f"{plot_prefix}test_loss_time",
        title=f"{objective_id} test loss over time",
        y_label="Test loss",
        runs=iterative_runs,
        primary_key="test_loss_x",
        secondary_key="test_loss_x_hat",
        baseline_runs=sklearn_runs,
        baseline_key="final_test_loss",
    )
    _maybe_add_time_plot(
        plot_root=plot_root,
        group_manifest=group_manifest,
        plot_id=f"{plot_prefix}test_accuracy_time",
        title=f"{objective_id} test accuracy over time",
        y_label="Test accuracy",
        runs=iterative_runs,
        primary_key="test_accuracy_x",
        secondary_key="test_accuracy_x_hat",
        baseline_runs=sklearn_runs,
        baseline_key="final_test_accuracy",
    )
    if iterative_runs:
        indices = _coordinate_indices_for_logreg(iterative_runs)
        panel_titles = ["Parameter norm"]
        panels: list[list[dict[str, Any]]] = [[]]
        for run in iterative_runs:
            run_label = _run_label(run)
            x_axis = _series_axis(len(run["x_norm"]))
            panels[0].append(
                _make_series(x_axis, run["x_norm"], label=f"{run_label} | x", source_label=run_label)
            )
            panels[0].append(
                _make_series(
                    x_axis,
                    run["x_hat_norm"],
                    label=f"{run_label} | x_hat",
                    style="dashed",
                    source_label=run_label,
                )
            )

        if baseline_vector is not None and panels[0]:
            baseline_axis = panels[0][0]["x"]
            baseline_norm = float(np.linalg.norm(np.asarray(baseline_vector, dtype=float)))
            for baseline_run in sklearn_runs:
                panels[0].append(
                    _make_constant_series(
                        baseline_axis,
                        baseline_norm,
                        label=f"{_run_label(baseline_run)} | baseline norm",
                    )
                )

        for coord_index in indices:
            coord_title = (
                f"Coordinate {coord_index} (bias)"
                if coord_index == int(iterative_runs[0]["objective_dimension"]) - 1
                else f"Coordinate {coord_index}"
            )
            panel_titles.append(coord_title)
            panel: list[dict[str, Any]] = []
            for run in iterative_runs:
                run_label = _run_label(run)
                x_axis, x, x_hat = _iterates_for_plot(run)
                panel.append(
                    _make_series(
                        x_axis,
                        x[:, coord_index],
                        label=f"{run_label} | x",
                        source_label=run_label,
                    )
                )
                panel.append(
                    _make_series(
                        x_axis,
                        x_hat[:, coord_index],
                        label=f"{run_label} | x_hat",
                        style="dashed",
                        source_label=run_label,
                    )
                )
            if baseline_vector is not None and panel:
                baseline_axis = panel[0]["x"]
                baseline_value = float(np.asarray(baseline_vector, dtype=float)[coord_index])
                for baseline_run in sklearn_runs:
                    panel.append(
                        _make_constant_series(
                            baseline_axis,
                            baseline_value,
                            label=f"{_run_label(baseline_run)} | baseline",
                        )
                    )
            panels.append(panel)

        _write_line_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}parameters_iterations",
            title=f"{objective_id} parameters over iterations",
            x_label="Iteration-aligned step",
            y_label="Value",
            panel_titles=panel_titles,
            panels=panels,
        )

        if baseline_vector is None:
            _record_skipped_plot(
                group_manifest,
                plot_id=f"{plot_prefix}distance_iterations",
                reason="sklearn baseline missing from saved runs",
            )
        else:
            distance_panels = [[]]
            for run in iterative_runs:
                run_label = _run_label(run)
                dist_axis, dist_x, dist_x_hat = _logreg_distance_series(run, baseline_vector)
                distance_panels[0].append(
                    _make_series(dist_axis, dist_x, label=f"{run_label} | x", source_label=run_label)
                )
                distance_panels[0].append(
                    _make_series(
                        dist_axis,
                        dist_x_hat,
                        label=f"{run_label} | x_hat",
                        style="dashed",
                        source_label=run_label,
                    )
                )
            _write_line_plot(
                plot_root=plot_root,
                group_manifest=group_manifest,
                plot_id=f"{plot_prefix}distance_iterations",
                title=f"{objective_id} distance to sklearn",
                x_label="Iteration",
                y_label="Distance to sklearn solution",
                panel_titles=[""],
                panels=distance_panels,
            )

    timed_runs = [run for run in runs if "total_runtime_seconds" in run]
    if timed_runs:
        _write_bar_plot(
            plot_root=plot_root,
            group_manifest=group_manifest,
            plot_id=f"{plot_prefix}runtime_summary",
            title=f"{objective_id} runtime by solver",
            x_label="Run",
            y_label="Runtime (s)",
            labels=[_run_label(run) for run in timed_runs],
            values=[float(run["total_runtime_seconds"]) for run in timed_runs],
        )
    else:
        _record_skipped_plot(
            group_manifest,
            plot_id=f"{plot_prefix}runtime_summary",
            reason="timing fields missing from saved runs",
        )


def _load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Results file must contain a JSON array: {path}")
    return [dict(run) for run in payload]


def _collect_results_paths(inputs: list[str]) -> list[Path]:
    roots = [Path(value) for value in inputs] if inputs else [Path("outputs")]
    discovered: set[Path] = set()
    for root in roots:
        if root.is_file():
            if root.name == "results.json":
                discovered.add(root.resolve())
            continue
        if not root.exists():
            continue
        for path in root.rglob("results.json"):
            discovered.add(path.resolve())
    return sorted(discovered)


def _write_manifest(plot_root: Path, manifest: dict[str, Any]) -> Path:
    plot_root.mkdir(parents=True, exist_ok=True)
    manifest_path = plot_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def process_results_file(results_path: Path) -> Path:
    runs = _load_results(results_path)
    schema = _detect_schema(runs)
    plot_root = results_path.parent / "plots"
    if plot_root.exists():
        shutil.rmtree(plot_root)
    manifest: dict[str, Any] = {
        "source": str(results_path),
        "schema": schema,
        "warnings": [],
        "groups": [],
    }

    if schema == "unknown":
        manifest["warnings"].append("Unrecognized results schema; no plots generated.")
        return _write_manifest(plot_root, manifest)

    partitions = _partition_runs(schema, runs)
    for partition_index, (partition_key, partition_runs) in enumerate(partitions, start=1):
        deduped_runs, duplicates = _deduplicate_runs(partition_runs)
        group_manifest: dict[str, Any] = {
            "group_id": f"partition_{partition_index:02d}",
            "group_key": partition_key,
            "objective_id": deduped_runs[0].get("objective_id"),
            "methods": [str(run["method"]) for run in deduped_runs],
            "deduplicated_runs": duplicates,
            "generated_plots": [],
            "skipped_plots": [],
        }
        plot_prefix = _partition_plot_prefix(deduped_runs[0], total_partitions=len(partitions))
        if schema == "bench":
            _plot_bench_partition(
                plot_root=plot_root,
                group_manifest=group_manifest,
                plot_prefix=plot_prefix,
                runs=deduped_runs,
            )
        else:
            _plot_logreg_partition(
                plot_root=plot_root,
                group_manifest=group_manifest,
                plot_prefix=plot_prefix,
                runs=deduped_runs,
            )
        manifest["groups"].append(group_manifest)

    return _write_manifest(plot_root, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PNG and HTML plot bundles for experiment results.json files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional results.json files or directories to scan. Default: recursively scan outputs/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_paths = _collect_results_paths(args.paths)
    if not results_paths:
        print("No results.json files found.")
        return 0

    for results_path in results_paths:
        try:
            manifest_path = process_results_file(results_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to generate plots for {results_path}: {exc}", file=sys.stderr)
            return 1
        print(f"Generated plots for {results_path} -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
