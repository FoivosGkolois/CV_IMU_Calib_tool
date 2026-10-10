"""Ticket 3: independent held-out replay and minimal GUI/API integration."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from computer.vision_imu_calibration import calibration_workflow as cw
from computer.vision_imu_calibration.imu_filter import KneeAngleFilterParameters
from computer.vision_imu_calibration.imu_serial import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB
from computer.vision_imu_calibration.runtime import RuntimeStatus, VisionRuntime
from computer.vision_imu_calibration.web_app import create_app


def _raw_row(index: int, angle: float, rate: float, *, start_ns: int) -> dict[str, str]:
    row = {
        "host_monotonic_ns": str(start_ns + index * 10_000_000),
        "schema_version": "1",
        "sample_index": str(index),
        "mcu_timestamp_us": str(index * 10_000),
        "thigh_ok": "true",
        "shank_ok": "true",
    }
    for prefix in ("thigh", "shank"):
        radians = math.radians(angle if prefix == "shank" else 0.0)
        accel = (0.0, -math.sin(radians), math.cos(radians))
        gyro = (rate if prefix == "shank" else 0.0, 0.0, 0.0)
        for axis, value in zip("xyz", accel):
            row[f"{prefix}_a{axis}_raw"] = str(round(value / ACCEL_G_PER_LSB))
        for axis, value in zip("xyz", gyro):
            row[f"{prefix}_g{axis}_raw"] = str(round(value / GYRO_DPS_PER_LSB))
    row.update({key: str(value) for key, value in cw._sample_from_row(row).to_record().items()
                if key.endswith(("_g", "_dps"))})
    return row


def _movement_data(
    reference_parameters: KneeAngleFilterParameters,
    *,
    start_ns: int,
    invalid_tracking: int = 0,
) -> cw.ArchiveData:
    raw_rows = []
    for index in range(1000):
        seconds = index * 0.01
        angle = 60.0 + 35.0 * math.sin(2.0 * math.pi * seconds / 2.5)
        rate = 35.0 * 2.0 * math.pi / 2.5 * math.cos(2.0 * math.pi * seconds / 2.5)
        raw_rows.append(_raw_row(index, angle, rate, start_ns=start_ns))
    estimates = cw.replay_filter(raw_rows, reference_parameters)
    sync_rows = []
    for position, index in enumerate(range(0, 1000, 10)):
        estimate = estimates[index]
        assert estimate.valid and estimate.knee_angle_deg is not None
        sync_rows.append({
            "host_monotonic_ns": raw_rows[index]["host_monotonic_ns"],
            "visual_knee_angle_deg": str(estimate.knee_angle_deg),
            "raw_knee_angle_deg": str(estimate.knee_angle_deg),
            "tracking_valid": "false" if position < invalid_tracking else "true",
            "tracked_leg": "right",
            "sync_delta_ms": "0",
            "imu_sample_index": str(index),
        })
    return cw.ArchiveData(raw_rows, sync_rows)


def _write_archive(path: Path, data: cw.ArchiveData, metadata: dict) -> Path:
    with ZipFile(path, "w") as archive:
        for name, rows in (("raw_imu.csv", data.raw_rows),
                           ("synchronized_timeline.csv", data.sync_rows)):
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(name, buffer.getvalue())
        archive.writestr("recording_metadata.json", json.dumps(metadata, sort_keys=True))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(base: KneeAngleFilterParameters) -> dict:
    return {
        "description": "subject A | right straps 1/2 | tripod floor mark A",
        "camera_profile": "fast",
        "camera_source": "camera:0",
        "pose_backend": "synthetic",
        "pose_tuning": {"leg_side": "right", "frame_rotation": "none"},
        "tracked_leg": "right",
        "imu_source": "COM4@921600",
        "imu_transport": "serial",
        "filter_parameters": cw._parameter_payload(base),
    }


def _make_completed_run(
    tmp_path: Path,
    *,
    setup_override: dict | None = None,
) -> tuple[Path, dict, dict[str, KneeAngleFilterParameters]]:
    run = tmp_path / "calibration_suggestions" / "calibration_run_20261009_120000"
    run.mkdir(parents=True)
    baseline = KneeAngleFilterParameters()
    static = replace(baseline, straight_leg_offset_deg=4.0, gyro_deadband_deg_s=.2,
                     thigh_gyro_bias_deg_s=.1, shank_gyro_bias_deg_s=-.1)
    bo = replace(static, alpha_stable=.55, alpha_dynamic=.72, max_accel_correction_deg=4.0)
    params = {"baseline": baseline, "static_calibrated": static, "bo_recommended": bo}
    setup = setup_override or _setup(baseline)
    training = _movement_data(static, start_ns=10_000_000_000)
    static_archive = _write_archive(
        run / cw.STATIC_ARCHIVE_FILENAME,
        training,
        {"recording_purpose": "vision_imu_filter_calibration",
         "calibration_test_type": "static", "calibration_setup": setup},
    )
    dynamic_archive = _write_archive(
        run / cw.DYNAMIC_ARCHIVE_FILENAME,
        training,
        {"recording_purpose": "vision_imu_filter_calibration",
         "calibration_test_type": "dynamic", "calibration_setup": setup},
    )
    report = {
        "schema_version": "vision_imu_calibration_report_v2",
        "status": "VALID",
        "method": "synchronized_static_dynamic_gp_bayesian_training",
        "source_archives": {"static": static_archive.name, "dynamic": dynamic_archive.name},
        "baseline_parameters": cw._parameter_payload(baseline),
        "static_suggestion": cw._parameter_payload(static),
        "dynamic_suggestion": cw._parameter_payload(bo),
        "recommended_parameters": cw._parameter_payload(bo),
        "calibration_setup": setup,
        "optimization": {
            "fixed_lag_ms": 0.0,
            "reference_sampling_policy": cw.HELDOUT_PAIRING_POLICY,
            "source_sha256": {"static": _sha(static_archive), "dynamic": _sha(dynamic_archive)},
        },
        # Deliberately state that BO was better during training. One test proves
        # that this does not mask worse independent validation.
        "metrics": {
            "dynamic_static_calibrated": {"rmse_deg": 8.0},
            "dynamic_recommended": {"rmse_deg": 4.0},
        },
    }
    (run / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run / cw.SUGGESTIONS_FILENAME).write_text("status: complete\n", encoding="utf-8")
    return run, setup, params


def _heldout_archive(
    tmp_path: Path,
    run: Path,
    setup: dict,
    reference_parameters: KneeAngleFilterParameters,
    *,
    invalid_tracking: int = 0,
    archived_setup: dict | None = None,
) -> Path:
    context = cw._load_heldout_context(run)
    validation_id = uuid.uuid4().hex
    metadata = {
        "recording_purpose": "vision_imu_filter_heldout_validation",
        "validation_id": validation_id,
        "calibration_run_id": run.name,
        "calibration_report_sha256": context["calibration_report_sha256"],
        "training_archive_sha256": context["training_sha256"],
        "parameter_snapshot_sha256": context["parameter_sha256"],
        "fixed_lag_ms": 0.0,
        "pairing_policy": cw.HELDOUT_PAIRING_POLICY,
        "calibration_setup": archived_setup if archived_setup is not None else setup,
    }
    return _write_archive(
        tmp_path / f"heldout_{validation_id}.zip",
        _movement_data(reference_parameters, start_ns=40_000_000_000,
                       invalid_tracking=invalid_tracking),
        metadata,
    )


def _poison(*args, **kwargs):
    raise AssertionError("Held-out validation attempted training or lag fitting")


def test_heldout_improvement_freezes_inputs_pairs_and_persists_across_restart(tmp_path, monkeypatch):
    run, setup, params = _make_completed_run(tmp_path)
    archive = _heldout_archive(tmp_path, run, setup, params["bo_recommended"])
    protected = {path.name: _sha(path) for path in (
        run / cw.STATIC_ARCHIVE_FILENAME,
        run / cw.DYNAMIC_ARCHIVE_FILENAME,
        run / "calibration_report.json",
        run / cw.SUGGESTIONS_FILENAME,
    )}
    context_before = cw._load_heldout_context(run)
    for name in ("optimize_filter_parameters", "_derive_static_parameters", "_estimate_lag_ms"):
        monkeypatch.setattr(cw, name, _poison)
    seen = []
    original = cw._evaluate_training_parameters

    def spy(data, parameters, pairs):
        seen.append((id(pairs), tuple(pairs), asdict(parameters)))
        return original(data, parameters, pairs)

    monkeypatch.setattr(cw, "_evaluate_training_parameters", spy)
    result = cw.validate_heldout_recording(
        run, archive, current_setup=setup, unchanged_setup=True
    )
    assert result["status"] == "IMPROVED_AGREEMENT"
    assert result["metrics"]["bo_recommended"]["rmse_deg"] < 1e-9
    assert result["metrics"]["static_calibrated"]["rmse_deg"] > 0
    assert {row[0] for row in seen} == {seen[0][0]}
    assert all(row[1] == seen[0][1] for row in seen)
    assert result["frozen_pair_count"] == 100
    assert result["fixed_lag_ms"] == 0.0
    assert result["frozen_parameters"] == context_before["parameter_payloads"]
    assert result["validation_archive"]["sha256"] not in context_before["training_sha256"].values()
    assert protected == {path.name: _sha(path) for path in (
        run / cw.STATIC_ARCHIVE_FILENAME,
        run / cw.DYNAMIC_ARCHIVE_FILENAME,
        run / "calibration_report.json",
        run / cw.SUGGESTIONS_FILENAME,
    )}
    history = list(run.glob("heldout_validation_*.json"))
    assert len(history) == 1
    repeated = cw.validate_heldout_recording(
        run, archive, current_setup=setup, unchanged_setup=True
    )
    assert repeated == result and list(run.glob("heldout_validation_*.json")) == history
    restored = cw.CalibrationWorkflow(run.parent)
    assert restored.status().state == "COMPLETE" and restored.status().valid
    assert restored.heldout_status()["status"] == "IMPROVED_AGREEMENT"


def test_training_improvement_but_worse_heldout_is_no_improvement(tmp_path):
    run, setup, params = _make_completed_run(tmp_path)
    archive = _heldout_archive(tmp_path, run, setup, params["static_calibrated"])
    result = cw.validate_heldout_recording(
        run, archive, current_setup=setup, unchanged_setup=True
    )
    assert result["training_rmse_deg"]["bo_recommended"] < result["training_rmse_deg"]["static_calibrated"]
    assert result["metrics"]["bo_recommended"]["rmse_deg"] > result["metrics"]["static_calibrated"]["rmse_deg"]
    assert result["status"] == "NO_IMPROVEMENT"


def test_setup_invalidation_is_persisted_and_blocks_restart_reuse(tmp_path):
    run, setup, _ = _make_completed_run(tmp_path)
    workflow = cw.CalibrationWorkflow(run.parent)
    assert workflow.status().valid

    invalidated = workflow.invalidate_setup("Pose configuration changed; recalibrate")
    marker = json.loads((run / cw.SETUP_INVALIDATION_FILENAME).read_text(encoding="utf-8"))
    assert invalidated.state == "INVALID" and not invalidated.valid
    assert marker["schema_version"] == "vision_imu_setup_invalidation_v1"
    assert marker["reason"] == "Pose configuration changed; recalibrate"

    restarted = cw.CalibrationWorkflow(run.parent)
    assert restarted.status().state == "INVALID"
    assert restarted.status().valid is False
    assert restarted.setup_metadata() == setup
    assert restarted.heldout_status() == {}


@pytest.mark.parametrize("failure", ["wrong_setup", "missing_setup", "dropout", "changed_archive_setup", "training_copy"])
def test_heldout_invalid_inputs_are_explicit_and_retryable(tmp_path, failure):
    run, setup, params = _make_completed_run(tmp_path)
    current_setup = setup
    if failure == "missing_setup":
        report_path = run / "calibration_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.pop("calibration_setup")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        archive = tmp_path / "missing.zip"
    elif failure == "training_copy":
        archive = tmp_path / "copied.zip"
        archive.write_bytes((run / cw.DYNAMIC_ARCHIVE_FILENAME).read_bytes())
    else:
        changed = dict(setup)
        changed["description"] = "different subject or camera mark"
        archive = _heldout_archive(
            tmp_path, run, setup, params["bo_recommended"],
            invalid_tracking=40 if failure == "dropout" else 0,
            archived_setup=changed if failure == "changed_archive_setup" else None,
        )
        if failure == "wrong_setup":
            current_setup = changed
    result = cw.validate_heldout_recording(
        run, archive, current_setup=current_setup, unchanged_setup=True
    )
    assert result["status"] == "INVALID" and result["error"]
    assert json.loads((run / cw.HELDOUT_REPORT_FILENAME).read_text(encoding="utf-8"))["status"] == "INVALID"


def test_operator_confirmation_is_required(tmp_path):
    run, setup, params = _make_completed_run(tmp_path)
    archive = _heldout_archive(tmp_path, run, setup, params["bo_recommended"])
    result = cw.validate_heldout_recording(
        run, archive, current_setup=setup, unchanged_setup=False
    )
    assert result["status"] == "INVALID"
    assert "confirmation" in result["error"]


class _ApiRuntime:
    def __init__(self, root: Path) -> None:
        self.recording_root = root / "recordings"
        self.calibration_root = root / "calibration"
        self.recording_root.mkdir()
        self.calibration_root.mkdir()
        self.started_setup = ""

    def start_recording(self, setup=""):
        self.started_setup = setup
        return {"active": True}

    def validate_latest_recording(self, *, setup, unchanged_setup):
        return {"status": "NO_IMPROVEMENT", "setup": setup,
                "unchanged_setup": unchanged_setup, "metrics": {}, "frozen_pair_count": 80}


def test_gui_and_api_use_existing_recorder_plus_one_validation_action(tmp_path):
    runtime = _ApiRuntime(tmp_path)
    client = create_app(runtime).test_client()  # type: ignore[arg-type]
    html = client.get("/").get_data(as_text=True)
    assert html.count('id="validateHeldout"') == 1
    assert 'id="startRecording"' in html and 'id="stopRecording"' in html
    assert "Validate finalized NEW trial" in html
    started = client.post("/api/recording/start", json={"setup": "same setup"})
    assert started.status_code == 200 and runtime.started_setup == "same setup"
    validated = client.post("/api/calibration/validate-heldout",
                            json={"setup": "same setup", "unchanged_setup": True})
    assert validated.status_code == 200
    assert validated.get_json()["status"] == "NO_IMPROVEMENT"


class _RuntimeCamera:
    description = "camera:0"
    camera_index = 0
    width = 640
    height = 480
    fps = 30.0

    def close(self):
        return None


class _RuntimeEstimator:
    description = "synthetic"


class _ReadySource:
    configured = True

    def status(self):
        return {
            "configured": True, "connected": True, "capture_active": True,
            "port": "COM4", "baud_rate": 921600, "transport": "serial",
            "state": "CAPTURING", "sample_rate_hz": 100.0, "sample_count": 1000,
            "parse_errors": 0, "sequence_gaps": 0, "queue_drops": 0,
            "thigh_ready": True, "shank_ready": True, "last_sample_index": 999,
            "last_error": "",
        }

    def stop(self):
        return None


def test_runtime_restores_stable_session_id_and_pose_invalidation(tmp_path):
    root = tmp_path / "restart"
    initial = VisionRuntime(
        _RuntimeCamera(), _RuntimeEstimator(),  # type: ignore[arg-type]
        recording_root=root / "recordings",
        calibration_root=root / "calibration_suggestions",
        imu_source=_ReadySource(),  # type: ignore[arg-type]
    )
    initial._status = RuntimeStatus(  # noqa: SLF001 - controlled setup snapshot
        state="RUNNING", source="camera:0", requested_leg="right", tracked_leg="right",
        tracking_valid=True, tracking_reason="ok", visual_knee_angle_deg=120.0,
        raw_knee_angle_deg=120.0, pose_confidence=.9, inference_fps=20.0,
        frame_index=10, error="",
    )
    setup = initial._calibration_setup_snapshot(  # noqa: SLF001
        "session_fixed_restart", initial.imu_source.status()
    )
    assert setup is not None
    _make_completed_run(root, setup_override=setup)

    restored = VisionRuntime(
        _RuntimeCamera(), _RuntimeEstimator(),  # type: ignore[arg-type]
        recording_root=root / "recordings",
        calibration_root=root / "calibration_suggestions",
        imu_source=_ReadySource(),  # type: ignore[arg-type]
    )
    assert restored.status()["calibration"]["state"] == "COMPLETE"
    assert restored.status()["calibration"]["session_setup_id"] == "session_fixed_restart"

    current = restored.tuning()
    preset = {key: current[key] for key in (
        "confidence_threshold", "landmark_smoothing", "angle_smoothing",
        "min_segment_length", "max_angle_jump_deg",
    )}
    preset["confidence_threshold"] = 0.41
    restored.save_tuning_preset(preset)

    restarted_again = VisionRuntime(
        _RuntimeCamera(), _RuntimeEstimator(),  # type: ignore[arg-type]
        recording_root=root / "recordings",
        calibration_root=root / "calibration_suggestions",
        imu_source=_ReadySource(),  # type: ignore[arg-type]
    )
    calibration = restarted_again.status()["calibration"]
    assert calibration["state"] == "INVALID" and not calibration["valid"]
    assert calibration["session_setup_id"] == "session_fixed_restart"
    assert restarted_again.calibration.heldout_status() == {}


def test_runtime_reuses_main_recorder_and_writes_immutable_heldout_metadata(tmp_path):
    root = tmp_path / "runtime"
    runtime = VisionRuntime(
        _RuntimeCamera(), _RuntimeEstimator(),  # type: ignore[arg-type]
        recording_root=root / "recordings",
        calibration_root=root / "calibration_suggestions",
        imu_source=_ReadySource(),  # type: ignore[arg-type]
    )
    runtime._status = RuntimeStatus(  # noqa: SLF001 - controlled integration state
        state="RUNNING", source="camera:0", requested_leg="right", tracked_leg="right",
        tracking_valid=True, tracking_reason="ok", visual_knee_angle_deg=120.0,
        raw_knee_angle_deg=120.0, pose_confidence=.9, inference_fps=20.0,
        frame_index=10, error="",
    )
    setup = runtime._calibration_setup_snapshot(  # noqa: SLF001
        "subject A | right straps 1/2 | tripod floor mark A",
        runtime.imu_source.status(),
    )
    assert setup is not None
    run, _, _ = _make_completed_run(root, setup_override=setup)
    runtime.calibration = cw.CalibrationWorkflow(root / "calibration_suggestions")
    recorder_identity = id(runtime.recorder)

    started = runtime.start_recording(setup=setup["description"])
    assert started["active"] is True and id(runtime.recorder) == recorder_identity
    stopped = runtime.stop_recording()
    archive = Path(stopped["archive_path"])
    assert archive.name.startswith("heldout_") and archive.is_file()
    metadata = cw._load_recording_metadata(archive)
    assert metadata["recording_purpose"] == "vision_imu_filter_heldout_validation"
    assert metadata["calibration_run_id"] == run.name
    assert metadata["calibration_setup"] == setup
    assert metadata["fixed_lag_ms"] == 0.0
