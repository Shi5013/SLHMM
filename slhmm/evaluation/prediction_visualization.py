"""Render cell observations, OSM truth, and Viterbi predictions."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re


os.environ.setdefault("MPLCONFIGDIR", "/tmp/slhmm-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from shapely.geometry import box


def _line_segments(geometries) -> list[np.ndarray]:
    segments = []
    for geometry in geometries:
        if geometry.is_empty:
            continue
        if geometry.geom_type == "LineString":
            segments.append(np.asarray(geometry.coords, dtype=float))
        elif geometry.geom_type == "MultiLineString":
            segments.extend(
                np.asarray(part.coords, dtype=float) for part in geometry.geoms
            )
    return segments


def _display_bounds(
    cell_xy: np.ndarray,
    route_geometries: list,
    minimum_buffer_m: float,
) -> tuple[float, float, float, float]:
    coordinates = [cell_xy]
    coordinates.extend(
        np.asarray(geometry.coords, dtype=float)
        for geometry in route_geometries
        if not geometry.is_empty and geometry.geom_type == "LineString"
    )
    points = np.concatenate(coordinates, axis=0)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    span = max(float(max_x - min_x), float(max_y - min_y))
    margin = max(float(minimum_buffer_m), 0.08 * span)
    return (
        float(min_x - margin),
        float(min_y - margin),
        float(max_x + margin),
        float(max_y + margin),
    )


def _background_geometries(matcher, bounds):
    indices = matcher.edge_index.query(box(*bounds))
    geometries = []
    seen = set()
    for index in indices:
        geometry = matcher.edge_geometries[int(index)]
        key = geometry.normalize().wkb
        if key not in seen:
            seen.add(key)
            geometries.append(geometry)
    return geometries


def _nice_scale_length(width_m: float) -> float:
    target = max(width_m / 5.0, 1.0)
    magnitude = 10.0 ** math.floor(math.log10(target))
    normalized = target / magnitude
    multiplier = 1.0 if normalized < 2.0 else 2.0 if normalized < 5.0 else 5.0
    return multiplier * magnitude


def _draw_scale_bar(axis, bounds) -> None:
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    length = _nice_scale_length(width)
    x0 = min_x + 0.06 * width
    y0 = min_y + 0.06 * height
    axis.plot(
        [x0, x0 + length],
        [y0, y0],
        color="#202020",
        linewidth=3.0,
        solid_capstyle="butt",
        zorder=10,
    )
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    axis.text(
        x0 + length / 2.0,
        y0 + 0.015 * height,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#202020",
        zorder=10,
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "trajectory"


def visualize_prediction(
    *,
    trajectory_id: str,
    alignment: dict,
    result: dict,
    cell_positions_metric,
    state_edges: list[tuple],
    matcher,
    output_dir: str | Path,
    minimum_buffer_m: float = 500.0,
    dpi: int = 180,
) -> Path:
    """Save one held-out trajectory prediction as a self-contained PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_ids = np.asarray(alignment["cell_ids"], dtype=np.int64)
    positions = cell_positions_metric[cell_ids]
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()
    cell_xy = np.asarray(positions, dtype=float)
    if cell_xy.ndim != 2 or cell_xy.shape[0] == 0 or cell_xy.shape[1] != 2:
        raise ValueError(f"Invalid cell positions for trajectory {trajectory_id}")

    truth_edges = [
        state_edges[int(state_id)] for state_id in alignment["truth_state_ids"]
    ]
    predicted_edges = [
        state_edges[int(state_id)]
        for state_id in result.get("expanded_state_ids", [])
    ]
    truth_lines = matcher.route_geometries_metric(truth_edges)
    predicted_lines = matcher.route_geometries_metric(predicted_edges)
    bounds = _display_bounds(
        cell_xy,
        [*truth_lines, *predicted_lines],
        minimum_buffer_m,
    )
    background = _background_geometries(matcher, bounds)

    figure, axis = plt.subplots(figsize=(11, 10), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#f7f7f5")
    background_segments = _line_segments(background)
    if background_segments:
        axis.add_collection(
            LineCollection(
                background_segments,
                colors="#b8bec5",
                linewidths=0.55,
                alpha=0.72,
                zorder=1,
            )
        )
    if truth_lines:
        axis.add_collection(
            LineCollection(
                _line_segments(truth_lines),
                colors="#1b9e77",
                linewidths=3.6,
                alpha=0.9,
                zorder=3,
            )
        )
    if predicted_lines:
        axis.add_collection(
            LineCollection(
                _line_segments(predicted_lines),
                colors="#d73027",
                linewidths=2.2,
                alpha=0.95,
                zorder=4,
            )
        )
    axis.plot(
        cell_xy[:, 0],
        cell_xy[:, 1],
        color="#2166ac",
        linewidth=1.5,
        linestyle="--",
        marker="o",
        markersize=4.3,
        markerfacecolor="#67a9cf",
        markeredgecolor="white",
        markeredgewidth=0.6,
        alpha=0.95,
        zorder=5,
    )
    axis.scatter(
        cell_xy[0, 0],
        cell_xy[0, 1],
        marker="^",
        s=95,
        color="#542788",
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )
    axis.scatter(
        cell_xy[-1, 0],
        cell_xy[-1, 1],
        marker="s",
        s=78,
        color="#f46d43",
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )

    min_x, min_y, max_x, max_y = bounds
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    _draw_scale_bar(axis, bounds)
    status = "success" if result.get("success") else result.get("reason", "failed")
    metric_summary = (
        f"LCS-F1={result.get('directed_edge_lcs_f1', 0.0):.3f} | "
        f"Road P/R={result.get('road_length_precision', 0.0):.3f}/"
        f"{result.get('road_length_recall', 0.0):.3f} | "
        f"RMF={result.get('rmf', 0.0):.3f} | "
        f"CMF50={result.get('cmf_50', 0.0):.3f}"
    )
    axis.set_title(
        f"Trajectory {trajectory_id} | decode={status}\n{metric_summary}",
        fontsize=13,
        pad=12,
    )
    legend = [
        Line2D([0], [0], color="#b8bec5", linewidth=1.2, label="OSM roads"),
        Line2D([0], [0], color="#1b9e77", linewidth=3.6, label="Ground truth"),
        Line2D([0], [0], color="#d73027", linewidth=2.2, label="Prediction"),
        Line2D(
            [0],
            [0],
            color="#2166ac",
            linestyle="--",
            marker="o",
            markersize=4,
            label="Cell trajectory",
        ),
        Line2D(
            [0],
            [0],
            color="#542788",
            marker="^",
            linestyle="none",
            markersize=8,
            label="Cell start",
        ),
        Line2D(
            [0],
            [0],
            color="#f46d43",
            marker="s",
            linestyle="none",
            markersize=7,
            label="Cell end",
        ),
    ]
    axis.legend(
        handles=legend,
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
    )

    output_path = output_dir / f"{_safe_filename(trajectory_id)}.png"
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.png")
    try:
        figure.savefig(
            temporary_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            format="png",
        )
        os.replace(temporary_path, output_path)
    finally:
        plt.close(figure)
    return output_path
