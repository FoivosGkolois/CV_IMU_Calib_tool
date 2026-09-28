from __future__ import annotations

import csv
from bisect import bisect_left
from pathlib import Path
from typing import Any


def build_synchronized_timeline(
    reference_csv_path: str | Path,
    imu_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    estimator_csv_path: str | Path | None = None,
) -> int:
    """Pair each camera sample with nearest raw-IMU and estimator samples in host time."""

    reference_path = Path(reference_csv_path)
    imu_path = Path(imu_csv_path)
    estimator_path = Path(estimator_csv_path) if estimator_csv_path is not None else None
    output_path = Path(output_csv_path)
    if not reference_path.is_file() or not imu_path.is_file():
        return 0

    references = _read_rows(reference_path)
    imu_rows = _read_rows(imu_path)
    estimator_rows = _read_rows(estimator_path) if estimator_path and estimator_path.is_file() else []
    if not references or not imu_rows:
        return 0

    imu_times = [int(row["host_monotonic_ns"]) for row in imu_rows]
    estimator_times = [int(row["host_monotonic_ns"]) for row in estimator_rows]
    fieldnames = list(references[0].keys()) + [
        "sync_delta_ms",
        *[f"imu_{key}" for key in imu_rows[0].keys()],
    ]
    if estimator_rows:
        fieldnames.extend(
            [
                "estimator_sync_delta_ms",
                *[f"estimator_{key}" for key in estimator_rows[0].keys()],
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for reference in references:
            reference_time = int(reference["host_monotonic_ns"])
            imu_index = _nearest_index(imu_times, reference_time)
            if imu_index is None:
                continue
            imu = imu_rows[imu_index]
            merged: dict[str, Any] = dict(reference)
            merged["sync_delta_ms"] = (imu_times[imu_index] - reference_time) / 1_000_000.0
            merged.update({f"imu_{key}": value for key, value in imu.items()})

            if estimator_rows:
                estimator_index = _nearest_index(estimator_times, reference_time)
                if estimator_index is not None:
                    estimator = estimator_rows[estimator_index]
                    merged["estimator_sync_delta_ms"] = (
                        estimator_times[estimator_index] - reference_time
                    ) / 1_000_000.0
                    merged.update(
                        {f"estimator_{key}": value for key, value in estimator.items()}
                    )
            writer.writerow(merged)
            written += 1
    return written


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nearest_index(times: list[int], target: int) -> int | None:
    if not times:
        return None
    insertion = bisect_left(times, target)
    candidates: list[int] = []
    if insertion > 0:
        candidates.append(insertion - 1)
    if insertion < len(times):
        candidates.append(insertion)
    if not candidates:
        return None
    return min(candidates, key=lambda index: abs(times[index] - target))
