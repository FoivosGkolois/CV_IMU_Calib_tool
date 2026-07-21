from __future__ import annotations

import json
import zipfile
from pathlib import Path

from computer.vision_imu_calibration.imu_serial import ImuCalibrationSample
from computer.vision_imu_calibration.uno_q_phase1_validation import (
    _package_result,
    evaluate_capture,
)


def sample(index: int, *, thigh_ok: bool = True, shank_ok: bool = True) -> ImuCalibrationSample:
    return ImuCalibrationSample(
        host_monotonic_ns=1_000_000_000 + index * 10_000_000,
        schema_version=1,
        sample_index=index,
        mcu_timestamp_us=100_000 + index * 10_000,
        thigh_ok=thigh_ok,
        shank_ok=shank_ok,
        thigh_ax_raw=0,
        thigh_ay_raw=0,
        thigh_az_raw=16384,
        thigh_gx_raw=0,
        thigh_gy_raw=0,
        thigh_gz_raw=0,
        shank_ax_raw=0,
        shank_ay_raw=0,
        shank_az_raw=16384,
        shank_gx_raw=0,
        shank_gy_raw=0,
        shank_gz_raw=0,
    )


def good_status() -> dict[str, object]:
    return {
        "thigh_ready": True,
        "shank_ready": True,
        "sequence_gaps": 0,
        "parse_errors": 0,
        "queue_drops": 0,
        "sample_rate_hz": 100.0,
    }


def test_phase1_evaluation_passes_exact_100_hz_contract() -> None:
    samples = [sample(index) for index in range(1000)]

    report = evaluate_capture(
        samples,
        requested_duration_s=10.0,
        final_status=good_status(),
    )

    assert report["passed"] is True
    assert report["sample_count"] == 1000
    assert report["effective_rate_hz"] == 100.0
    assert report["mcu_period_ms"]["median"] == 10.0
    assert all(check["passed"] for check in report["checks"])


def test_phase1_evaluation_rejects_gap_and_invalid_sample() -> None:
    samples = [sample(index) for index in range(500)]
    samples[250] = sample(252, thigh_ok=False)
    status = good_status()
    status["sequence_gaps"] = 2

    report = evaluate_capture(
        samples,
        requested_duration_s=5.0,
        final_status=status,
    )

    assert report["passed"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "sequence_continuity" in failed
    assert "thigh_sample_validity" in failed


def test_phase1_packaging_creates_one_zip_without_loose_files(tmp_path: Path) -> None:
    samples = [sample(index) for index in range(500)]
    report = evaluate_capture(
        samples,
        requested_duration_s=5.0,
        final_status=good_status(),
    )

    archive_path = _package_result(
        output_root=tmp_path,
        timestamp="20260721_120000",
        samples=samples,
        report=report,
    )

    suite_dir = archive_path.parent
    assert [path.name for path in suite_dir.iterdir()] == [archive_path.name]
    with zipfile.ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "phase1_samples.csv",
            "phase1_validation_report.json",
        ]
        loaded = json.loads(archive.read("phase1_validation_report.json"))
    assert loaded["passed"] is True


def test_uno_q_sketch_preserves_imucal_v1_contract() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    sketch = (
        repository_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "arduino"
        / "uno_q_dual_imu_stream"
        / "uno_q_dual_imu_stream.ino"
    ).read_text(encoding="utf-8")

    assert "#if !defined(ARDUINO_UNO_Q)" in sketch
    assert "#define IMUCAL_I2C_BUS Wire1" in sketch
    assert "SERIAL_BAUD_RATE = 921600" in sketch
    assert "SAMPLE_PERIOD_US = 10000" in sketch
    assert "THIGH_IMU_ADDRESS = 0x6A" in sketch
    assert "SHANK_IMU_ADDRESS = 0x6B" in sketch
    assert "SCHEMA_VERSION = 1" in sketch
    assert 'Serial.print("IMUCAL,")' in sketch
    assert 'firmware=uno_q_dual_imu_stream_v1' in sketch
    assert "motor" not in sketch.lower()
