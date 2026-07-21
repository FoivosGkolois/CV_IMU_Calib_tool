from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .imu_serial import ImuCalibrationSample, SerialImuSource, list_serial_ports

UINT32_MODULUS = 1 << 32
TARGET_SAMPLE_PERIOD_MS = 10.0


@dataclass(frozen=True)
class Phase1Thresholds:
    minimum_effective_rate_hz: float = 90.0
    maximum_effective_rate_hz: float = 110.0
    minimum_sample_fraction: float = 0.90
    minimum_valid_fraction: float = 1.0
    minimum_mean_accel_norm_g: float = 0.50
    maximum_mean_accel_norm_g: float = 1.50
    minimum_median_period_ms: float = 9.5
    maximum_median_period_ms: float = 10.5
    maximum_p95_period_ms: float = 11.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(percentile))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _u32_deltas(values: Sequence[int]) -> list[int]:
    return [
        (int(current) - int(previous)) % UINT32_MODULUS
        for previous, current in zip(values, values[1:])
    ]


def _sequence_gap_count(samples: Sequence[ImuCalibrationSample]) -> int:
    gaps = 0
    for previous, current in zip(samples, samples[1:]):
        difference = int(current.sample_index) - int(previous.sample_index)
        if difference > 1:
            gaps += difference - 1
        elif difference <= 0:
            gaps += 1
    return gaps


def _mean_accel_norm_g(samples: Sequence[ImuCalibrationSample], prefix: str) -> float:
    norms: list[float] = []
    for sample in samples:
        record = sample.to_record()
        ax = float(record[f"{prefix}_ax_g"])
        ay = float(record[f"{prefix}_ay_g"])
        az = float(record[f"{prefix}_az_g"])
        norms.append(math.sqrt(ax * ax + ay * ay + az * az))
    return statistics.fmean(norms) if norms else math.nan


def evaluate_capture(
    samples: Sequence[ImuCalibrationSample],
    *,
    requested_duration_s: float,
    final_status: dict[str, Any],
    thresholds: Phase1Thresholds | None = None,
) -> dict[str, Any]:
    """Evaluate whether an UNO Q capture preserves the established IMUCAL contract."""

    thresholds = thresholds or Phase1Thresholds()
    count = len(samples)
    expected_minimum = max(
        50,
        int(math.floor(float(requested_duration_s) * 100.0 * thresholds.minimum_sample_fraction)),
    )

    thigh_valid_fraction = (
        sum(1 for sample in samples if sample.thigh_ok) / count if count else 0.0
    )
    shank_valid_fraction = (
        sum(1 for sample in samples if sample.shank_ok) / count if count else 0.0
    )

    mcu_deltas_ms = [
        delta / 1000.0
        for delta in _u32_deltas([sample.mcu_timestamp_us for sample in samples])
    ]
    median_period_ms = statistics.median(mcu_deltas_ms) if mcu_deltas_ms else math.nan
    p95_period_ms = _percentile(mcu_deltas_ms, 0.95)
    maximum_period_ms = max(mcu_deltas_ms) if mcu_deltas_ms else math.nan

    if len(samples) >= 2:
        host_elapsed_s = (
            samples[-1].host_monotonic_ns - samples[0].host_monotonic_ns
        ) / 1_000_000_000.0
        effective_rate_hz = (len(samples) - 1) / host_elapsed_s if host_elapsed_s > 0 else 0.0
    else:
        host_elapsed_s = 0.0
        effective_rate_hz = 0.0

    independent_sequence_gaps = _sequence_gap_count(samples)
    thigh_accel_norm_g = _mean_accel_norm_g(samples, "thigh")
    shank_accel_norm_g = _mean_accel_norm_g(samples, "shank")

    checks = [
        {
            "name": "both_imus_ready",
            "passed": bool(final_status.get("thigh_ready"))
            and bool(final_status.get("shank_ready")),
            "value": {
                "thigh_ready": bool(final_status.get("thigh_ready")),
                "shank_ready": bool(final_status.get("shank_ready")),
            },
            "requirement": "Both IMUs must report ready.",
        },
        {
            "name": "enough_samples",
            "passed": count >= expected_minimum,
            "value": count,
            "requirement": f"At least {expected_minimum} samples.",
        },
        {
            "name": "effective_rate_hz",
            "passed": thresholds.minimum_effective_rate_hz
            <= effective_rate_hz
            <= thresholds.maximum_effective_rate_hz,
            "value": effective_rate_hz,
            "requirement": (
                f"{thresholds.minimum_effective_rate_hz:.1f}-"
                f"{thresholds.maximum_effective_rate_hz:.1f} Hz."
            ),
        },
        {
            "name": "mcu_period_median_ms",
            "passed": bool(mcu_deltas_ms)
            and thresholds.minimum_median_period_ms
            <= median_period_ms
            <= thresholds.maximum_median_period_ms,
            "value": median_period_ms,
            "requirement": (
                f"{thresholds.minimum_median_period_ms:.1f}-"
                f"{thresholds.maximum_median_period_ms:.1f} ms."
            ),
        },
        {
            "name": "mcu_period_p95_ms",
            "passed": bool(mcu_deltas_ms)
            and p95_period_ms <= thresholds.maximum_p95_period_ms,
            "value": p95_period_ms,
            "requirement": f"At most {thresholds.maximum_p95_period_ms:.1f} ms.",
        },
        {
            "name": "sequence_continuity",
            "passed": independent_sequence_gaps == 0
            and int(final_status.get("sequence_gaps", 0)) == 0,
            "value": {
                "independent_gaps": independent_sequence_gaps,
                "source_gaps": int(final_status.get("sequence_gaps", 0)),
            },
            "requirement": "Zero missing or non-monotonic sample indices.",
        },
        {
            "name": "parser_integrity",
            "passed": int(final_status.get("parse_errors", 0)) == 0,
            "value": int(final_status.get("parse_errors", 0)),
            "requirement": "Zero malformed IMUCAL frames.",
        },
        {
            "name": "host_queue_integrity",
            "passed": int(final_status.get("queue_drops", 0)) == 0,
            "value": int(final_status.get("queue_drops", 0)),
            "requirement": "Zero host queue drops.",
        },
        {
            "name": "thigh_sample_validity",
            "passed": thigh_valid_fraction >= thresholds.minimum_valid_fraction,
            "value": thigh_valid_fraction,
            "requirement": "Every collected thigh sample must be valid.",
        },
        {
            "name": "shank_sample_validity",
            "passed": shank_valid_fraction >= thresholds.minimum_valid_fraction,
            "value": shank_valid_fraction,
            "requirement": "Every collected shank sample must be valid.",
        },
        {
            "name": "thigh_acceleration_plausibility",
            "passed": thresholds.minimum_mean_accel_norm_g
            <= thigh_accel_norm_g
            <= thresholds.maximum_mean_accel_norm_g,
            "value": thigh_accel_norm_g,
            "requirement": (
                f"Mean norm {thresholds.minimum_mean_accel_norm_g:.2f}-"
                f"{thresholds.maximum_mean_accel_norm_g:.2f} g."
            ),
        },
        {
            "name": "shank_acceleration_plausibility",
            "passed": thresholds.minimum_mean_accel_norm_g
            <= shank_accel_norm_g
            <= thresholds.maximum_mean_accel_norm_g,
            "value": shank_accel_norm_g,
            "requirement": (
                f"Mean norm {thresholds.minimum_mean_accel_norm_g:.2f}-"
                f"{thresholds.maximum_mean_accel_norm_g:.2f} g."
            ),
        },
    ]

    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "contract": {
            "transport": "USB serial",
            "baud_rate": 921600,
            "frame_prefix": "IMUCAL",
            "schema_version": 1,
            "target_sample_rate_hz": 100,
        },
        "requested_duration_s": float(requested_duration_s),
        "sample_count": count,
        "host_elapsed_s": host_elapsed_s,
        "effective_rate_hz": effective_rate_hz,
        "mcu_period_ms": {
            "median": median_period_ms,
            "p95": p95_period_ms,
            "maximum": maximum_period_ms,
        },
        "valid_fraction": {
            "thigh": thigh_valid_fraction,
            "shank": shank_valid_fraction,
        },
        "mean_acceleration_norm_g": {
            "thigh": thigh_accel_norm_g,
            "shank": shank_accel_norm_g,
        },
        "checks": checks,
        "final_source_status": dict(final_status),
        "thresholds": asdict(thresholds),
    }


def _choose_port(explicit_port: str | None) -> str:
    if explicit_port:
        return explicit_port.strip()

    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("No serial ports were found.")

    print("Available serial ports:")
    for index, port in enumerate(ports, start=1):
        description = port.get("description", "")
        print(f"  {index}. {port['device']}  {description}")

    while True:
        answer = input("Select the UNO Q port number: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            selected = 0
        if 1 <= selected <= len(ports):
            return ports[selected - 1]["device"]
        print("Invalid selection.")


def _write_samples_csv(path: Path, samples: Iterable[ImuCalibrationSample]) -> None:
    rows = [sample.to_record() for sample in samples]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _package_result(
    *,
    output_root: Path,
    timestamp: str,
    samples: Sequence[ImuCalibrationSample],
    report: dict[str, Any],
) -> Path:
    suite_dir = output_root / f"uno_q_phase1_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=False)
    archive_path = suite_dir / f"uno_q_phase1_validation_{timestamp}.zip"

    temporary_dir = Path(tempfile.mkdtemp(prefix="uno_q_phase1_"))
    try:
        report_path = temporary_dir / "phase1_validation_report.json"
        sample_path = temporary_dir / "phase1_samples.csv"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_samples_csv(sample_path, samples)

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(report_path, arcname=report_path.name)
            archive.write(sample_path, arcname=sample_path.name)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return archive_path


def run_validation(
    *,
    port: str,
    duration_s: float,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    source = SerialImuSource(port, queue_size=4096)
    samples: list[ImuCalibrationSample] = []
    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")

    source.start()
    try:
        if not source.wait_until_connected(timeout_s=8.0):
            status = source.status()
            raise RuntimeError(
                f"UNO Q did not answer on {port}: {status.get('last_error') or status.get('state')}"
            )

        source.request_status(timeout_s=2.0)
        status_deadline = time.monotonic() + 2.0
        initial_status = source.status()
        while time.monotonic() < status_deadline:
            initial_status = source.status()
            if initial_status.get("thigh_ready") and initial_status.get("shank_ready"):
                break
            time.sleep(0.05)
        if not initial_status.get("thigh_ready") or not initial_status.get("shank_ready"):
            raise RuntimeError(
                "Both IMUs must report ready before capture. "
                f"Status: thigh={initial_status.get('thigh_ready')} "
                f"shank={initial_status.get('shank_ready')}"
            )

        source.start_capture(timeout_s=2.0)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            sample = source.get_sample(timeout_s=0.20)
            if sample is not None:
                samples.append(sample)

        source.stop_capture(timeout_s=2.0)
        while True:
            sample = source.get_sample(timeout_s=0.0)
            if sample is None:
                break
            samples.append(sample)

        final_status = source.status()
        report = evaluate_capture(
            samples,
            requested_duration_s=duration_s,
            final_status=final_status,
        )
        report.update(
            {
                "tool": "UNO Q phase 1 IMUCAL migration validation",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "port": port,
                "initial_source_status": initial_status,
            }
        )
    finally:
        source.stop()

    archive_path = _package_result(
        output_root=output_root,
        timestamp=timestamp,
        samples=samples,
        report=report,
    )
    return archive_path, report


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the UNO Q MCU port against the established IMUCAL v1 serial, "
            "timing, readiness, and continuity contract."
        )
    )
    parser.add_argument("--port", help="UNO Q USB serial port, for example COM4.")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Capture duration in seconds (default: 15).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("diagnostic_logs"),
        help="Folder that will receive one dedicated suite folder containing one ZIP.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.duration < 5.0:
        parser.error("--duration must be at least 5 seconds")

    try:
        port = _choose_port(args.port)
        print(f"Connecting to UNO Q on {port}...")
        archive_path, report = run_validation(
            port=port,
            duration_s=float(args.duration),
            output_root=args.output_root,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2

    print("\n=== UNO Q phase 1 validation ===")
    print(f"Result: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Samples: {report['sample_count']}")
    print(f"Effective rate: {report['effective_rate_hz']:.2f} Hz")
    print(f"MCU period median: {report['mcu_period_ms']['median']:.4f} ms")
    print(f"Sequence gaps: {report['final_source_status'].get('sequence_gaps', 0)}")
    print(f"Parse errors: {report['final_source_status'].get('parse_errors', 0)}")
    print(f"Queue drops: {report['final_source_status'].get('queue_drops', 0)}")
    print(f"ZIP: {archive_path.resolve()}")

    if not report["passed"]:
        print("Failed checks:")
        for check in report["checks"]:
            if not check["passed"]:
                print(f"  - {check['name']}: {check['value']} ({check['requirement']})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
