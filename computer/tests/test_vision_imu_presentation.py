from __future__ import annotations

import csv
import json
from pathlib import Path

from computer.vision_imu_calibration.presentation import build_acquisition_presentation


def test_builds_descriptive_summary_and_overview(tmp_path: Path) -> None:
    synchronized = tmp_path / "sync.csv"
    summary = tmp_path / "summary.json"
    plot = tmp_path / "overview.png"
    fields = [
        "recording_time_s",
        "tracking_valid",
        "visual_knee_angle_deg",
        "sync_delta_ms",
        "estimator_sync_delta_ms",
        "estimator_filter_valid",
        "estimator_knee_angle_deg",
    ]
    with synchronized.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"recording_time_s": 0.0, "tracking_valid": True, "visual_knee_angle_deg": 10, "sync_delta_ms": 1, "estimator_sync_delta_ms": 1, "estimator_filter_valid": True, "estimator_knee_angle_deg": 9},
                {"recording_time_s": 1.0, "tracking_valid": True, "visual_knee_angle_deg": 50, "sync_delta_ms": -2, "estimator_sync_delta_ms": -2, "estimator_filter_valid": True, "estimator_knee_angle_deg": 48},
            ]
        )
    result = build_acquisition_presentation(synchronized, summary, plot)
    assert result["synchronized_rows"] == 2
    assert result["visual_valid_fraction"] == 1.0
    assert result["imu_estimator_valid_fraction"] == 1.0
    assert summary.is_file()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert "no scoring" in payload["scope"]
    assert plot.is_file()
