from __future__ import annotations

from pathlib import Path

from computer.vision_imu_calibration.camera_calibration import (
    CameraCalibrationSolveResult,
    CheckerboardSpecification,
)
from computer.vision_imu_calibration.marker_reference import CameraCalibration
from computer.vision_imu_calibration.reference_setup import ReferenceSetupStore


def _result() -> CameraCalibrationSolveResult:
    calibration = CameraCalibration.from_values(
        [[800.0, 0.0, 320.0], [0.0, 805.0, 240.0], [0.0, 0.0, 1.0]],
        [0.01, -0.02, 0.0, 0.0, 0.0],
        image_width=640,
        image_height=480,
    )
    return CameraCalibrationSolveResult(
        calibration=calibration,
        rms_reprojection_error_px=0.42,
        mean_view_error_px=0.35,
        max_view_error_px=0.78,
        accepted_views=15,
    )


def test_reference_setup_round_trip_and_log(tmp_path: Path) -> None:
    store = ReferenceSetupStore(tmp_path / "reference_setup")
    store.save_camera_calibration(
        _result(),
        camera_description="USB camera 0",
        camera_fps=30.0,
        tag_size_m=0.070,
        thigh_tag_id=0,
        shank_tag_id=1,
        specification=CheckerboardSpecification(),
    )

    loaded = store.load_camera_calibration(expected_width=640, expected_height=480)
    assert loaded.valid is True
    assert loaded.calibration is not None
    assert loaded.calibration.image_width == 640
    assert loaded.rms_reprojection_error_px == 0.42
    assert store.yaml_path.is_file()
    assert "CAMERA_CALIBRATION_SAVED" in store.log_path.read_text(encoding="utf-8")


def test_reference_setup_rejects_resolution_mismatch(tmp_path: Path) -> None:
    store = ReferenceSetupStore(tmp_path / "reference_setup")
    store.save_camera_calibration(
        _result(),
        camera_description="USB camera 0",
        camera_fps=30.0,
        tag_size_m=0.070,
        thigh_tag_id=0,
        shank_tag_id=1,
        specification=CheckerboardSpecification(),
    )
    loaded = store.load_camera_calibration(expected_width=1280, expected_height=720)
    assert loaded.valid is False
    assert loaded.state == "RESOLUTION_MISMATCH"
    assert loaded.calibration is None


def test_reference_setup_rejects_corrupt_yaml(tmp_path: Path) -> None:
    store = ReferenceSetupStore(tmp_path / "reference_setup")
    store.yaml_path.write_text("camera: [broken", encoding="utf-8")
    loaded = store.load_camera_calibration(expected_width=640, expected_height=480)
    assert loaded.valid is False
    assert loaded.state == "INVALID"
    assert "invalid" in loaded.message.lower()


def test_reference_setup_persists_known_angle_validation_and_snapshot(tmp_path: Path) -> None:
    store = ReferenceSetupStore(tmp_path / "reference_setup")
    store.save_camera_calibration(
        _result(),
        camera_description="USB camera 0",
        camera_fps=30.0,
        tag_size_m=0.070,
        thigh_tag_id=0,
        shank_tag_id=1,
        specification=CheckerboardSpecification(),
    )
    validation = {
        "status": "passed",
        "validated_utc": "2026-09-29T12:00:00+00:00",
        "metrics": {
            "passed": True,
            "rmse_deg": 1.2,
            "max_repeatability_sd_deg": 0.6,
        },
    }
    first = store.save_marker_reference_validation(validation)
    second = store.save_marker_reference_validation(validation)

    assert first["reference_setup_id"] == second["reference_setup_id"]
    assert store.marker_reference_validation()["metrics"]["passed"] is True
    before_sha = store.current_sha256()
    snapshot = store.snapshot_to(tmp_path / "run" / "reference_setup_snapshot.yaml")
    assert snapshot["sha256"] == before_sha
    assert (tmp_path / "run" / "reference_setup_snapshot.yaml").read_bytes() == store.yaml_path.read_bytes()
    assert "MARKER_REFERENCE_VALIDATION_PASSED" in store.log_path.read_text(encoding="utf-8")
