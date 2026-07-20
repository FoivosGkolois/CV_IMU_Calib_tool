from __future__ import annotations

import csv
from bisect import bisect_left
from pathlib import Path


def build_synchronized_timeline(
    pose_csv_path: str | Path,
    imu_csv_path: str | Path,
    output_csv_path: str | Path,
) -> int:
    """Pair each camera sample with the nearest serial IMU sample in host time."""

    pose_path = Path(pose_csv_path)
    imu_path = Path(imu_csv_path)
    output_path = Path(output_csv_path)
    if not pose_path.is_file() or not imu_path.is_file():
        return 0

    with pose_path.open("r", newline="", encoding="utf-8") as handle:
        poses = list(csv.DictReader(handle))
    with imu_path.open("r", newline="", encoding="utf-8") as handle:
        imu_rows = list(csv.DictReader(handle))
    if not poses or not imu_rows:
        return 0

    imu_times = [int(row["host_monotonic_ns"]) for row in imu_rows]
    fieldnames = list(poses[0].keys()) + [
        "sync_delta_ms",
        *[f"imu_{key}" for key in imu_rows[0].keys()],
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pose in poses:
            pose_time = int(pose["host_monotonic_ns"])
            insertion = bisect_left(imu_times, pose_time)
            candidates = []
            if insertion > 0:
                candidates.append(insertion - 1)
            if insertion < len(imu_rows):
                candidates.append(insertion)
            if not candidates:
                continue
            best_index = min(candidates, key=lambda index: abs(imu_times[index] - pose_time))
            imu = imu_rows[best_index]
            merged = dict(pose)
            merged["sync_delta_ms"] = (imu_times[best_index] - pose_time) / 1_000_000.0
            merged.update({f"imu_{key}": value for key, value in imu.items()})
            writer.writerow(merged)
            written += 1
    return written
