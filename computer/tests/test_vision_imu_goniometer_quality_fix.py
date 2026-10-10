"""Fidelity checks for guided static holds at Fast and Quality profiles."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from computer.vision_imu_calibration.reference_validation import (
    evaluate_static_samples, guided_capture_report,
)


def make_capture(*, fps: float = 4.5, invalid_indices: set[int] | None = None,
                 drifting: bool = False) -> list[dict]:
    n = int(round(3 * fps))
    result = []
    for i in range(n):
        jitter = (0.5 if i % 2 else -0.5)
        result.append({
            "host_monotonic_ns": int((100 + i / fps) * 1e9),
            "valid": i not in (invalid_indices or set()),
            "tracked_leg": "left",
            "raw_knee_angle_deg": 90 + jitter * 2 + (12 * i / max(1, n-1) if drifting else 0),
            "visual_knee_angle_deg": 90 + jitter + (12 * i / max(1, n-1) if drifting else 0),
            # Deliberate >2% landmark displacement, must not define angular stillness.
            "hip_x": 0.41 + 0.028 * (i % 2), "hip_y": 0.22,
            "knee_x": 0.43, "knee_y": 0.50,
            "ankle_x": 0.48, "ankle_y": 0.76,
        })
    return result


def synthetic_recording(path: Path, rows: list[dict]) -> None:
    content = io.StringIO()
    writer = csv.DictWriter(content, fieldnames=["recording_time_s", "tracking_valid", "tracked_leg",
                                                      "raw_knee_angle_deg", "visual_knee_angle_deg"])
    writer.writeheader()
    start_ns = rows[0]["host_monotonic_ns"]
    for row in rows:
        writer.writerow({"recording_time_s": (row["host_monotonic_ns"] - start_ns) / 1e9,
                         "tracking_valid": row["valid"], "tracked_leg": row["tracked_leg"],
                         "raw_knee_angle_deg": row["raw_knee_angle_deg"],
                         "visual_knee_angle_deg": row["visual_knee_angle_deg"]})
    with ZipFile(path, "w") as z:
        z.writestr("pose_timeline.csv", content.getvalue())
        z.writestr("recording_metadata.json", json.dumps({"source": "quality-synthetic"}))


def test_quality_fps_capture_is_accepted_with_natural_jitter(tmp_path):
    rows = make_capture(fps=4.5)
    hold = evaluate_static_samples(rows, start_ns=int(100e9), end_ns=int(103e9))
    assert hold["accepted"], hold["reason"]
    assert hold["frames_valid"] >= 8
    assert hold["stability_deg"]["raw_knee_angle_deg"]["full_range_deg"] >= 2
    path = tmp_path / "quality.zip"
    synthetic_recording(path, rows)
    report = guided_capture_report(path, {"reference_angle_deg": 90.0,
                                          "profile": {"camera": {"width": 1280, "height": 720}},
                                          "hold_start_host_monotonic_ns": int(100e9),
                                          "hold_end_host_monotonic_ns": int(103e9)}, rows)
    assert report["status"] == "Measured"
    assert report["hold"]["criteria"]["min_valid_frames"] == 8
    assert report["metrics"]["raw"]["n"] >= 8
    assert report["metrics"]["filtered"]["sd_deg"] > 0


def test_quality_tolerates_isolated_missing_pose_observations():
    rows = make_capture(fps=4.5, invalid_indices={4, 9})
    hold = evaluate_static_samples(rows, start_ns=int(100e9), end_ns=int(103e9))
    assert hold["accepted"], hold["reason"]
    assert .70 <= hold["coverage"] < 1.0


def test_quality_rejects_insufficient_processed_frames():
    rows = make_capture(fps=2.0)
    hold = evaluate_static_samples(rows)
    assert not hold["accepted"]
    assert "Too few" in hold["reason"]


def test_quality_rejects_real_movement_and_missing_coverage():
    rows = make_capture(fps=4.5, drifting=True)
    assert not evaluate_static_samples(rows)["accepted"]
    rows = make_capture(fps=4.5, invalid_indices={4, 5, 6, 7, 8})
    assert not evaluate_static_samples(rows)["accepted"]


def test_profile_change_during_capture_is_blocked_without_reconfiguring(tmp_path):
    from computer.vision_imu_calibration.runtime import VisionRuntime

    class Camera:
        description = "camera:0"
        width, height, fps = 640, 480, 30
        video_path = None
        camera_index = 0
        def reconfigure(self, **kwargs):
            raise AssertionError("Camera must not be reconfigured mid-capture")

    class Estimator:
        description = "fake pose"

    runtime = VisionRuntime(Camera(), Estimator(), recording_root=tmp_path / "records",
                            calibration_root=tmp_path / "cal")
    runtime._reference["state"] = "CAPTURING"
    with pytest.raises(RuntimeError, match="goniometer capture"):
        runtime.update_camera_profile("quality")


def test_quality_profile_real_runtime_capture_pipeline(tmp_path):
    """End-to-end 720p video, 90-degree rotation, 3s hold, recorder/export.

    Uses deterministic synthetic pose and camera, NOT a claim of UNO Q hardware
    acceptance or of inference-model accuracy at 1280x720.
    """
    import math
    import time
    import numpy as np
    cv2 = pytest.importorskip("cv2")
    from computer.vision_imu_calibration.models import Landmark, PoseFrame
    from computer.vision_imu_calibration.runtime import VisionRuntime

    class QualityCamera:
        description = "synthetic-camera-1280x720"
        width, height, fps = 1280, 720, 30.0
        camera_index = 0
        active_camera_index = 0
        video_path = None
        def open(self):
            pass
        def read(self):
            time.sleep(.19)  # ~5 FPS incoming frames
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        def close(self):
            pass

    class SyntheticEstimator:
        description = "synthetic-90-deg-pose"
        def open(self):
            pass
        def close(self):
            pass
        def estimate(self, frame, timestamp_ms):
            aspect = frame.shape[1] / frame.shape[0]
            # With image aspect scaling, this is exactly a 90-degree knee.
            def p(x, y):
                return Landmark(x, y, 1, 1)
            return PoseFrame(timestamp_ms, {"right_hip": p(.5, .22),
                                             "right_knee": p(.5, .5),
                                             "right_ankle": p(.5 + .2 / aspect, .5)})

    runtime = VisionRuntime(QualityCamera(), SyntheticEstimator(),
                            recording_root=tmp_path / "recordings", calibration_root=tmp_path / "cal")
    runtime.update_tuning({"leg_side": "left", "frame_rotation": "counterclockwise_90"})
    assert runtime.camera_status()["profile"] == "quality"
    runtime.start()
    try:
        deadline = time.monotonic() + 8
        while not runtime.status()["tracking_valid"] and time.monotonic() < deadline:
            time.sleep(.05)
        assert runtime.status()["tracking_valid"], f"reason={runtime._status.tracking_reason} status={runtime._status}"
        runtime.reference_action("start_session", {})
        runtime.reference_action("capture", {})
        deadline = time.monotonic() + 12
        while runtime.reference_status()["state"] in {"CAPTURING", "FINALIZING"} and time.monotonic() < deadline:
            time.sleep(.05)
        result = runtime.reference_status()
        assert result["state"] == "MEASURED", result["message"]
        assert result["result"]["metrics"]["frames_total"] >= 8
        assert result["result"]["hold"]["frames_valid"] >= 8
        assert result["result"]["metrics"]["raw"]["mean_deg"] == pytest.approx(90.0, abs=.1)
        assert result["guided"]["index"] == 1
    finally:
        runtime.stop()
