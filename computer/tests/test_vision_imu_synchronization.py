from __future__ import annotations

import csv
from pathlib import Path

from computer.vision_imu_calibration.synchronization import build_synchronized_timeline


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_nearest_host_timestamp_alignment(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"
    imu = tmp_path / "imu.csv"
    out = tmp_path / "sync.csv"
    write_csv(
        pose,
        ["host_monotonic_ns", "frame_index", "visual_knee_angle_deg"],
        [
            {"host_monotonic_ns": 1_000_000_000, "frame_index": 1, "visual_knee_angle_deg": 10},
            {"host_monotonic_ns": 1_020_000_000, "frame_index": 2, "visual_knee_angle_deg": 20},
        ],
    )
    write_csv(
        imu,
        ["host_monotonic_ns", "sample_index", "mcu_timestamp_us"],
        [
            {"host_monotonic_ns": 995_000_000, "sample_index": 10, "mcu_timestamp_us": 100},
            {"host_monotonic_ns": 1_015_000_000, "sample_index": 11, "mcu_timestamp_us": 200},
            {"host_monotonic_ns": 1_025_000_000, "sample_index": 12, "mcu_timestamp_us": 300},
        ],
    )

    assert build_synchronized_timeline(pose, imu, out) == 2
    with out.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["imu_sample_index"] == "10"
    assert float(rows[0]["sync_delta_ms"]) == -5.0
    assert rows[1]["imu_sample_index"] == "11"
    assert float(rows[1]["sync_delta_ms"]) == -5.0


def test_empty_imu_timeline_produces_no_sync_rows(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"
    imu = tmp_path / "imu.csv"
    out = tmp_path / "sync.csv"
    write_csv(pose, ["host_monotonic_ns"], [{"host_monotonic_ns": 1}])
    write_csv(imu, ["host_monotonic_ns"], [])
    assert build_synchronized_timeline(pose, imu, out) == 0
    assert not out.exists()


def test_synchronization_includes_processed_estimator_timeline(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"
    imu = tmp_path / "imu.csv"
    estimator = tmp_path / "estimator.csv"
    out = tmp_path / "sync.csv"
    write_csv(
        pose,
        ["host_monotonic_ns", "recording_time_s", "tracking_valid", "visual_knee_angle_deg"],
        [{"host_monotonic_ns": 1_000_000_000, "recording_time_s": 0.1, "tracking_valid": True, "visual_knee_angle_deg": 40}],
    )
    write_csv(
        imu,
        ["host_monotonic_ns", "sample_index"],
        [{"host_monotonic_ns": 1_001_000_000, "sample_index": 7}],
    )
    write_csv(
        estimator,
        ["host_monotonic_ns", "sample_index", "filter_valid", "knee_angle_deg"],
        [{"host_monotonic_ns": 1_001_000_000, "sample_index": 7, "filter_valid": True, "knee_angle_deg": 38.5}],
    )
    assert build_synchronized_timeline(pose, imu, out, estimator_csv_path=estimator) == 1
    with out.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["estimator_sample_index"] == "7"
    assert float(row["estimator_knee_angle_deg"]) == 38.5
    assert float(row["estimator_sync_delta_ms"]) == 1.0
