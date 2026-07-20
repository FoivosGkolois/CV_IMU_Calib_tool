from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def build_acquisition_presentation(
    synchronized_csv_path: str | Path,
    summary_json_path: str | Path,
    overview_plot_path: str | Path,
) -> dict[str, Any]:
    """Create descriptive acquisition evidence without parameter optimization."""

    sync_path = Path(synchronized_csv_path)
    summary_path = Path(summary_json_path)
    plot_path = Path(overview_plot_path)
    rows = _read_rows(sync_path)

    visual_values = _finite_values(rows, "visual_knee_angle_deg")
    imu_values = _finite_values(rows, "estimator_knee_angle_deg")
    sync_abs = [abs(value) for value in _finite_values(rows, "sync_delta_ms")]
    estimator_sync_abs = [
        abs(value) for value in _finite_values(rows, "estimator_sync_delta_ms")
    ]
    visual_valid_count = sum(_truthy(row.get("tracking_valid")) for row in rows)
    estimator_valid_count = sum(_truthy(row.get("estimator_filter_valid")) for row in rows)
    times = _finite_values(rows, "recording_time_s")

    summary: dict[str, Any] = {
        "schema_version": "vision_imu_acquisition_summary_v1",
        "synchronized_rows": len(rows),
        "duration_s": (max(times) - min(times)) if len(times) >= 2 else 0.0,
        "visual_valid_rows": visual_valid_count,
        "visual_valid_fraction": visual_valid_count / len(rows) if rows else 0.0,
        "imu_estimator_valid_rows": estimator_valid_count,
        "imu_estimator_valid_fraction": estimator_valid_count / len(rows) if rows else 0.0,
        "visual_angle_range_deg": _range_or_none(visual_values),
        "imu_angle_range_deg": _range_or_none(imu_values),
        "camera_to_raw_imu_sync_abs_ms": _distribution(sync_abs),
        "camera_to_estimator_sync_abs_ms": _distribution(estimator_sync_abs),
        "scope": "descriptive acquisition and presentation only; no scoring, tuning, or recommendation",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_overview_plot(rows, plot_path)
    return summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        text = row.get(key)
        if text in {None, ""}:
            continue
        try:
            value = float(text)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _range_or_none(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values), "span": max(values) - min(values)}


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "max": max(ordered),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def _write_overview_plot(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    time_values: list[float] = []
    visual: list[float] = []
    imu: list[float] = []
    for row in rows:
        try:
            time_s = float(row.get("recording_time_s", ""))
        except (TypeError, ValueError):
            continue
        time_values.append(time_s)
        visual.append(_float_or_nan(row.get("visual_knee_angle_deg")))
        imu.append(_float_or_nan(row.get("estimator_knee_angle_deg")))

    if not time_values:
        return
    figure = plt.figure(figsize=(11, 5.5))
    axis = figure.add_subplot(111)
    axis.plot(time_values, visual, label="Visual knee flexion")
    axis.plot(time_values, imu, label="Current IMU filter output")
    axis.set_xlabel("Recording time [s]")
    axis.set_ylabel("Knee flexion [deg]")
    axis.set_title("Vision–IMU acquisition overview")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan
