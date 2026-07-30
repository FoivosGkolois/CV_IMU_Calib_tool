from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from pathlib import Path

import pytest
import yaml

from computer.vision_imu_calibration.calibration_workflow import (
    CalibrationWorkflow,
    analyze_calibration_run,
)
from computer.vision_imu_calibration.imu_filter import (
    KneeAngleFilterParameters,
)
from computer.vision_imu_calibration.imu_serial import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB
from computer.vision_imu_calibration.recorder import AnnotatedVideoRecorder


def _raw_row(index: int, *, host_ns: int, flexion_deg: float, flexion_rate_dps: float) -> dict[str, object]:
    shank_ay_g = -math.sin(math.radians(flexion_deg))
    shank_az_g = math.cos(math.radians(flexion_deg))
    return {
        "recording_time_s": index * 0.01,
        "host_monotonic_ns": host_ns,
        "schema_version": 1,
        "sample_index": index,
        "mcu_timestamp_us": index * 10_000,
        "thigh_ok": True,
        "shank_ok": True,
        "thigh_ax_raw": 0,
        "thigh_ay_raw": 0,
        "thigh_az_raw": round(1.0 / ACCEL_G_PER_LSB),
        "thigh_gx_raw": 0,
        "thigh_gy_raw": 0,
        "thigh_gz_raw": 0,
        "shank_ax_raw": 0,
        "shank_ay_raw": round(shank_ay_g / ACCEL_G_PER_LSB),
        "shank_az_raw": round(shank_az_g / ACCEL_G_PER_LSB),
        "shank_gx_raw": round(flexion_rate_dps / GYRO_DPS_PER_LSB),
        "shank_gy_raw": 0,
        "shank_gz_raw": 0,
        "thigh_ax_g": 0.0,
        "thigh_ay_g": 0.0,
        "thigh_az_g": 1.0,
        "thigh_gx_dps": 0.0,
        "thigh_gy_dps": 0.0,
        "thigh_gz_dps": 0.0,
        "shank_ax_g": 0.0,
        "shank_ay_g": shank_ay_g,
        "shank_az_g": shank_az_g,
        "shank_gx_dps": flexion_rate_dps,
        "shank_gy_dps": 0.0,
        "shank_gz_dps": 0.0,
    }


def _archive(path: Path, *, dynamic: bool, tracking_valid: bool = True) -> Path:
    count = 1000 if dynamic else 400
    start_ns = 8_000_000_000
    raw_rows: list[dict[str, object]] = []
    flexions: list[float] = []
    for index in range(count):
        t = index * 0.01
        flexion = 40.0 + 30.0 * math.sin(2.0 * math.pi * t / 2.5) if dynamic else 30.0
        rate = 30.0 * (2.0 * math.pi / 2.5) * math.cos(2.0 * math.pi * t / 2.5) if dynamic else 0.0
        flexions.append(flexion)
        raw_rows.append(
            _raw_row(index, host_ns=start_ns + index * 10_000_000, flexion_deg=flexion, flexion_rate_dps=rate)
        )

    sync_rows: list[dict[str, object]] = []
    for index in range(0, count, 10):
        sync_rows.append(
            {
                "recording_time_s": index * 0.01,
                "host_monotonic_ns": start_ns + index * 10_000_000,
                "frame_index": index // 10,
                "visual_knee_angle_deg": 180.0 - flexions[index],
                "tracking_valid": tracking_valid,
                "tracking_reason": "ok" if tracking_valid else "low_confidence",
                "sync_delta_ms": 0.0,
                "imu_sample_index": index,
            }
        )

    def csv_bytes(rows: list[dict[str, object]], fields: list[str]) -> bytes:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "raw_imu.csv",
            csv_bytes(raw_rows, AnnotatedVideoRecorder.IMU_FIELDS),
        )
        archive.writestr(
            "synchronized_timeline.csv",
            csv_bytes(sync_rows, list(sync_rows[0].keys())),
        )
    return path


def test_analysis_exports_loadable_recommended_settings(tmp_path: Path) -> None:
    static_archive = _archive(tmp_path / "static.zip", dynamic=False)
    dynamic_archive = _archive(tmp_path / "dynamic.zip", dynamic=True)
    output = tmp_path / "calibration_run"

    result = analyze_calibration_run(
        static_archive,
        dynamic_archive,
        base_parameters=KneeAngleFilterParameters(),
        output_directory=output,
    )

    assert result["valid"] is True
    assert result["recommended_parameters"]["relative_angle_sign"] == pytest.approx(1.0)
    recommendation = Path(result["recommendation_path"])
    report_path = Path(result["report_path"])
    overview = Path(result["overview_path"])
    assert recommendation.is_file()
    assert report_path.is_file()
    assert overview.is_file()
    loaded = KneeAngleFilterParameters.load_yaml(recommendation)
    loaded.validate()
    suggestions = yaml.safe_load(recommendation.read_text(encoding="utf-8"))
    assert suggestions["status"] == "complete"
    assert suggestions["source_trials"] == {
        "static": "static_trial.zip",
        "dynamic": "dynamic_trial.zip",
    }
    assert suggestions["static_stage"]["suggestion"]
    assert suggestions["dynamic_stage"]["suggestion"]
    assert suggestions["final_recommended_settings"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "VALID"
    assert report["metrics"]["candidate_count"] > 1
    assert report["metrics"]["static_recommended"]["lag_ms"] == pytest.approx(0.0)
    assert report["metrics"]["dynamic_recommended"]["pair_count"] >= 80


def test_invalid_tracking_does_not_export_recommendation(tmp_path: Path) -> None:
    static_archive = _archive(tmp_path / "static.zip", dynamic=False, tracking_valid=False)
    dynamic_archive = _archive(tmp_path / "dynamic.zip", dynamic=True)
    output = tmp_path / "invalid_run"

    result = analyze_calibration_run(
        static_archive,
        dynamic_archive,
        base_parameters=KneeAngleFilterParameters(),
        output_directory=output,
    )

    assert result["valid"] is False
    recommendation = Path(result["recommendation_path"])
    assert recommendation.name == "imu_filter_suggestions.yaml"
    assert recommendation.is_file()
    suggestions = yaml.safe_load(recommendation.read_text(encoding="utf-8"))
    assert suggestions["status"] == "invalid"
    assert suggestions["final_recommended_settings"] is None
    report = json.loads((output / "calibration_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "INVALID"
    assert "static_tracking_fraction" in report["failed_gates"]


def test_workflow_requires_static_before_dynamic(tmp_path: Path) -> None:
    workflow = CalibrationWorkflow(tmp_path)
    with pytest.raises(RuntimeError, match="static test"):
        workflow.prepare_test("dynamic", base_parameters=KneeAngleFilterParameters())
    prepared = workflow.prepare_test("static", base_parameters=KneeAngleFilterParameters())
    assert prepared["duration_s"] == pytest.approx(12.0)
    assert workflow.status().active is True
    assert Path(prepared["run_directory"]).is_dir()


def test_dynamic_test_rejects_changed_filter_after_static(tmp_path: Path) -> None:
    workflow = CalibrationWorkflow(tmp_path)
    base = KneeAngleFilterParameters()
    prepared = workflow.prepare_test("static", base_parameters=base)
    static_archive = _archive(
        Path(prepared["run_directory"]) / "static_trial.zip", dynamic=False
    )
    workflow.finish_test("static", static_archive)

    changed = KneeAngleFilterParameters(alpha_stable=0.85)
    with pytest.raises(RuntimeError, match="changed after the static test"):
        workflow.prepare_test("dynamic", base_parameters=changed)


def test_one_calibration_run_updates_one_shared_yaml_across_both_stages(
    tmp_path: Path,
) -> None:
    workflow = CalibrationWorkflow(tmp_path / "calibration_suggestions")
    base = KneeAngleFilterParameters()
    prepared = workflow.prepare_test("static", base_parameters=base)
    run_directory = Path(prepared["run_directory"])
    assert prepared["archive_name"] == "static_trial.zip"

    static_archive = _archive(run_directory / prepared["archive_name"], dynamic=False)
    static_status = workflow.finish_test("static", static_archive)
    shared_yaml = Path(static_status.recommendation_path)
    static_payload = yaml.safe_load(shared_yaml.read_text(encoding="utf-8"))
    assert static_status.state == "AWAITING_DYNAMIC"
    assert static_payload["status"] == "awaiting_dynamic"
    assert static_payload["static_stage"]["suggestion"]
    assert static_payload["dynamic_stage"]["suggestion"] is None

    prepared_dynamic = workflow.prepare_test("dynamic", base_parameters=base)
    assert Path(prepared_dynamic["run_directory"]) == run_directory
    assert prepared_dynamic["archive_name"] == "dynamic_trial.zip"
    dynamic_archive = _archive(
        run_directory / prepared_dynamic["archive_name"], dynamic=True
    )
    final_status = workflow.finish_test("dynamic", dynamic_archive)
    final_payload = yaml.safe_load(shared_yaml.read_text(encoding="utf-8"))
    assert final_status.state == "COMPLETE"
    assert Path(final_status.recommendation_path) == shared_yaml
    assert final_payload["status"] == "complete"
    assert final_payload["static_stage"]["suggestion"]
    assert final_payload["dynamic_stage"]["suggestion"]
    assert final_payload["final_recommended_settings"]
    assert (run_directory / "static_trial.zip").is_file()
    assert (run_directory / "dynamic_trial.zip").is_file()


def test_dynamic_failure_keeps_static_stage_and_shared_yaml_retryable(tmp_path: Path) -> None:
    workflow = CalibrationWorkflow(tmp_path / "calibration_suggestions")
    base = KneeAngleFilterParameters()
    prepared = workflow.prepare_test("static", base_parameters=base)
    run_directory = Path(prepared["run_directory"])
    static_archive = _archive(run_directory / "static_trial.zip", dynamic=False)
    workflow.finish_test("static", static_archive)

    workflow.prepare_test("dynamic", base_parameters=base)
    failed = workflow.fail("camera disconnected", stage="dynamic")
    payload = yaml.safe_load(
        (run_directory / "imu_filter_suggestions.yaml").read_text(encoding="utf-8")
    )

    assert failed.state == "AWAITING_DYNAMIC"
    assert failed.static_suggestion
    assert failed.dynamic_suggestion == {}
    assert payload["status"] == "awaiting_dynamic"
    assert payload["static_stage"]["status"] == "valid"
    assert payload["dynamic_stage"]["status"] == "pending"
    assert payload["last_error"] == "camera disconnected"
    assert not (run_directory / "dynamic_trial.zip").exists()


def test_static_failure_marks_shared_yaml_without_fake_suggestion(tmp_path: Path) -> None:
    workflow = CalibrationWorkflow(tmp_path / "calibration_suggestions")
    prepared = workflow.prepare_test("static", base_parameters=KneeAngleFilterParameters())
    run_directory = Path(prepared["run_directory"])

    failed = workflow.fail("camera unavailable", stage="static")
    payload = yaml.safe_load(
        (run_directory / "imu_filter_suggestions.yaml").read_text(encoding="utf-8")
    )

    assert failed.state == "FAILED"
    assert payload["status"] == "failed_static"
    assert payload["static_stage"]["status"] == "invalid"
    assert payload["static_stage"]["suggestion"] is None
    assert payload["last_error"] == "camera unavailable"
