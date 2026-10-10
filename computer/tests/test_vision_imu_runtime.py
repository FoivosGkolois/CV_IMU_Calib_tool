from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from computer.vision_imu_calibration.geometry import KneeAngleResult
from computer.vision_imu_calibration.models import Landmark, PoseFrame
from computer.vision_imu_calibration.runtime import RuntimeStatus, VisionRuntime


class BlockingCamera:
    description = "blocking-camera"

    def __init__(self) -> None:
        self.release_open = threading.Event()
        self.closed = False

    def open(self) -> None:
        self.release_open.wait(timeout=1.0)

    def read(self):
        return None

    def close(self) -> None:
        self.closed = True
        self.release_open.set()


class FakeEstimator:
    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def estimate(self, frame, timestamp_ms):
        return None


def test_runtime_start_returns_before_slow_camera_initialization(tmp_path: Path) -> None:
    camera = BlockingCamera()
    runtime = VisionRuntime(camera, FakeEstimator(), recording_root=tmp_path)  # type: ignore[arg-type]
    started = time.perf_counter()
    runtime.start()
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2
    assert runtime.status()["state"] == "STARTING"
    camera.release_open.set()
    runtime.stop()


def test_angle_filter_smooths_valid_measurements() -> None:
    first = KneeAngleResult(20.0, True, 0.9, "ok")
    filtered, state = VisionRuntime._filter_angle(
        first,
        previous=10.0,
        smoothing=0.5,
        max_jump_deg=45.0,
    )
    assert filtered.valid
    assert filtered.angle_deg == pytest.approx(15.0)
    assert state == pytest.approx(15.0)


def test_angle_filter_rejects_large_jump() -> None:
    result = KneeAngleResult(100.0, True, 0.9, "ok")
    filtered, state = VisionRuntime._filter_angle(
        result,
        previous=20.0,
        smoothing=0.2,
        max_jump_deg=30.0,
    )
    assert not filtered.valid
    assert filtered.reason == "angle_jump"
    assert state == pytest.approx(20.0)


class PreviewCamera:
    description = "preview-camera"

    def __init__(self) -> None:
        import numpy as np

        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.closed = False

    def open(self) -> None:
        return None

    def read(self):
        time.sleep(0.01)
        return self.frame.copy()

    def close(self) -> None:
        self.closed = True


class FixedObliquePoseEstimator:
    """Return a 90-degree knee angle in pixel space on any frame shape."""

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def estimate(self, frame, timestamp_ms: int) -> PoseFrame:
        height, width = frame.shape[:2]
        def point(x: float, y: float) -> Landmark:
            return Landmark(x=x, y=y, visibility=1.0, presence=1.0)
        return PoseFrame(
            timestamp_ms=timestamp_ms,
            landmarks={
                # The runtime's RIGHT leg resolves to the MoveNet LEFT chain.
                "left_hip": point(0.5 + 30.0 / width, 0.5 - 30.0 / height),
                "left_knee": point(0.5, 0.5),
                "left_ankle": point(0.5 + 30.0 / width, 0.5 + 30.0 / height),
            },
        )


def test_runtime_corrects_raw_and_filtered_angles_for_camera_aspect_ratio(tmp_path: Path) -> None:
    runtime = VisionRuntime(
        PreviewCamera(), FixedObliquePoseEstimator(), recording_root=tmp_path
    )  # type: ignore[arg-type]
    runtime.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = runtime.status()
            if status["tracking_valid"]:
                assert status["raw_knee_angle_deg"] == pytest.approx(90.0, abs=1e-7)
                assert status["visual_knee_angle_deg"] == pytest.approx(90.0, abs=1e-7)
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"No valid pose measurement: {runtime.status()}")
    finally:
        runtime.stop()


class BlockingEstimator:
    def __init__(self) -> None:
        self.release_open = threading.Event()
        self.closed = False

    def open(self) -> None:
        self.release_open.wait(timeout=2.0)

    def close(self) -> None:
        self.closed = True
        self.release_open.set()

    def estimate(self, frame, timestamp_ms):
        return None


class FailingEstimator:
    def open(self) -> None:
        raise RuntimeError("pose backend unavailable")

    def close(self) -> None:
        return None

    def estimate(self, frame, timestamp_ms):
        raise AssertionError("estimate must not run when open failed")


def test_runtime_publishes_camera_preview_before_pose_model_is_ready(tmp_path: Path) -> None:
    camera = PreviewCamera()
    estimator = BlockingEstimator()
    runtime = VisionRuntime(camera, estimator, recording_root=tmp_path)  # type: ignore[arg-type]
    runtime.start()

    sequence, jpeg = runtime.wait_for_jpeg(0, timeout_s=1.0)
    status = runtime.status()

    assert sequence >= 1
    assert jpeg is not None
    assert status["state"] == "CAMERA_PREVIEW"
    assert status["frame_index"] >= 1
    assert status["tracking_reason"] == "initializing_pose_model"

    estimator.release_open.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and runtime.status()["state"] != "RUNNING":
        time.sleep(0.01)
    assert runtime.status()["state"] == "RUNNING"
    runtime.stop()


def test_runtime_preserves_camera_preview_when_pose_model_open_fails(tmp_path: Path) -> None:
    camera = PreviewCamera()
    runtime = VisionRuntime(camera, FailingEstimator(), recording_root=tmp_path)  # type: ignore[arg-type]
    runtime.start()

    sequence, jpeg = runtime.wait_for_jpeg(0, timeout_s=1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and runtime.status()["state"] != "AI_FAILED":
        time.sleep(0.01)
    status = runtime.status()

    assert sequence >= 1
    assert jpeg is not None
    assert status["state"] == "AI_FAILED"
    assert status["frame_index"] >= 1
    assert status["tracking_reason"] == "pose_model_failed"
    assert "pose backend unavailable" in status["error"]
    runtime.stop()


class FakeRotationCv2:
    ROTATE_90_CLOCKWISE = 0
    ROTATE_90_COUNTERCLOCKWISE = 1
    ROTATE_180 = 2

    @staticmethod
    def rotate(frame, code):
        return (frame, code)


def test_runtime_rotates_portrait_frames_using_selected_orientation() -> None:
    from computer.vision_imu_calibration.tuning import FrameRotation

    cv2 = FakeRotationCv2()
    assert VisionRuntime._rotate_frame("frame", FrameRotation.NONE, cv2) == "frame"
    assert VisionRuntime._rotate_frame("frame", FrameRotation.CLOCKWISE_90, cv2) == ("frame", 0)
    assert VisionRuntime._rotate_frame("frame", FrameRotation.COUNTERCLOCKWISE_90, cv2) == ("frame", 1)
    assert VisionRuntime._rotate_frame("frame", FrameRotation.ROTATE_180, cv2) == ("frame", 2)


class ManualImuSource:
    instances: list["ManualImuSource"] = []

    def __init__(self, port: str, *, baud_rate: int) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.started = False
        self.stopped = False
        self.capture_active = False
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_sample(self, timeout_s: float = 0.1):
        time.sleep(min(timeout_s, 0.01))
        return None

    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "connected": self.started and not self.stopped,
            "capture_active": self.capture_active,
            "port": self.port,
            "baud_rate": self.baud_rate,
            "state": "CONNECTED" if self.started and not self.stopped else "STOPPED",
            "sample_rate_hz": 0.0,
            "sample_count": 0,
            "parse_errors": 0,
            "sequence_gaps": 0,
            "queue_drops": 0,
            "thigh_ready": False,
            "shank_ready": False,
            "last_sample_index": None,
            "last_error": "",
        }


class ReconfigurableCamera(PreviewCamera):
    video_path = None
    width = 640
    height = 480
    fps = 30.0

    def profile(self):
        return {"width": self.width, "height": self.height, "fps": self.fps}

    def reconfigure(self, *, width: int, height: int, fps: float) -> None:
        self.width = width
        self.height = height
        self.fps = fps


def test_runtime_manual_imu_connect_and_disconnect(tmp_path: Path) -> None:
    ManualImuSource.instances.clear()
    runtime = VisionRuntime(
        PreviewCamera(),
        FakeEstimator(),
        recording_root=tmp_path,
        imu_source_factory=ManualImuSource,
    )  # type: ignore[arg-type]

    connected = runtime.connect_imu("COM7", baud_rate=921600)
    assert connected["port"] == "COM7"
    assert connected["connected"] is True
    source = ManualImuSource.instances[-1]

    disconnected = runtime.disconnect_imu()
    assert disconnected["state"] == "DISCONNECTED"
    assert source.stopped is True


def test_runtime_camera_profile_changes_live_source_settings(tmp_path: Path) -> None:
    camera = ReconfigurableCamera()
    runtime = VisionRuntime(camera, FakeEstimator(), recording_root=tmp_path)  # type: ignore[arg-type]
    runtime._status = RuntimeStatus(
        state="RUNNING", source=camera.description, requested_leg="auto", tracked_leg=None,
        tracking_valid=False, tracking_reason="test", visual_knee_angle_deg=None,
        raw_knee_angle_deg=None, pose_confidence=0.0, inference_fps=0.0,
        frame_index=0, error="",
    )

    result = runtime.update_camera_profile("quality")

    assert result["profile"] == "quality"
    assert result["width"] == 1280
    assert result["height"] == 720
    assert result["fps"] == 30.0


class StreamingManualImuSource(ManualImuSource):
    def wait_until_connected(self, timeout_s: float = 3.0) -> bool:
        return self.started and not self.stopped

    def start_capture(self, timeout_s: float = 1.5) -> None:
        self.capture_active = True


def test_manual_connect_starts_continuous_calibration_stream(tmp_path: Path) -> None:
    StreamingManualImuSource.instances.clear()
    runtime = VisionRuntime(
        PreviewCamera(),
        FakeEstimator(),
        recording_root=tmp_path,
        imu_source_factory=StreamingManualImuSource,
    )  # type: ignore[arg-type]
    status = runtime.connect_imu("COM8")
    assert status["capture_active"] is True
    runtime.disconnect_imu()


def test_acquisition_history_is_bounded_and_decimated(tmp_path: Path) -> None:
    runtime = VisionRuntime(PreviewCamera(), FakeEstimator(), recording_root=tmp_path)  # type: ignore[arg-type]
    now = time.perf_counter_ns()
    with runtime._imu_filter_lock:
        for index in range(100):
            runtime._acquisition_history.append(
                {
                    "host_monotonic_ns": now - (99 - index) * 10_000_000,
                    "visual_knee_angle_deg": float(index),
                    "visual_valid": True,
                    "imu_knee_angle_deg": float(index - 1),
                    "imu_valid": True,
                    "angle_difference_deg": 1.0,
                }
            )
    rows = runtime.acquisition_history(seconds=2.0, max_points=20)
    assert len(rows) <= 20
    assert rows[-1]["angle_difference_deg"] == 1.0


def test_imu_loop_updates_live_filter_estimate(tmp_path: Path) -> None:
    from computer.vision_imu_calibration.imu_serial import ImuCalibrationSample

    class OneSampleSource:
        def __init__(self) -> None:
            self.sent = False

        def get_sample(self, timeout_s: float = 0.1):
            if self.sent:
                time.sleep(0.01)
                return None
            self.sent = True
            return ImuCalibrationSample(
                host_monotonic_ns=1_000_000_000,
                schema_version=1,
                sample_index=1,
                mcu_timestamp_us=10_000,
                thigh_ok=True,
                shank_ok=True,
                thigh_ax_raw=0,
                thigh_ay_raw=0,
                thigh_az_raw=16393,
                thigh_gx_raw=0,
                thigh_gy_raw=0,
                thigh_gz_raw=0,
                shank_ax_raw=0,
                shank_ay_raw=-16393,
                shank_az_raw=0,
                shank_gx_raw=0,
                shank_gy_raw=0,
                shank_gz_raw=0,
            )

        def status(self):
            return {"configured": True, "capture_active": True}

    runtime = VisionRuntime(PreviewCamera(), FakeEstimator(), recording_root=tmp_path)  # type: ignore[arg-type]
    with runtime._imu_lock:
        runtime._imu_source = OneSampleSource()  # type: ignore[assignment]
    thread = threading.Thread(target=runtime._imu_loop)
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not runtime.imu_estimator_status()["valid"]:
        time.sleep(0.01)
    runtime._stop_event.set()
    thread.join(timeout=1.0)
    status = runtime.imu_estimator_status()
    assert status["valid"] is True
    assert status["knee_angle_deg"] == pytest.approx(90.0, abs=0.2)


class ReadyCalibrationImuSource:
    configured = True

    def status(self):
        return {
            "configured": True,
            "connected": True,
            "capture_active": True,
            "port": "COM4",
            "baud_rate": 921600,
            "state": "CAPTURING",
            "sample_rate_hz": 100.0,
            "sample_count": 100,
            "parse_errors": 0,
            "sequence_gaps": 0,
            "queue_drops": 0,
            "thigh_ready": True,
            "shank_ready": True,
            "last_sample_index": 99,
            "last_error": "",
        }

    def stop(self):
        return None


def test_runtime_starts_guided_static_calibration_only_when_ready(tmp_path: Path) -> None:
    runtime = VisionRuntime(
        PreviewCamera(),
        FakeEstimator(),
        recording_root=tmp_path / "video_recordings",
        calibration_root=tmp_path / "calibration_suggestions",
        imu_source=ReadyCalibrationImuSource(),  # type: ignore[arg-type]
    )  # type: ignore[arg-type]
    runtime._status = RuntimeStatus(  # noqa: SLF001 - focused runtime state test
        state="RUNNING",
        source="preview-camera",
        requested_leg="right",
        tracked_leg="right",
        tracking_valid=True,
        tracking_reason="ok",
        visual_knee_angle_deg=150.0,
        raw_knee_angle_deg=150.0,
        pose_confidence=0.9,
        inference_fps=10.0,
        frame_index=10,
        error="",
    )

    status = runtime.start_calibration_test("static")
    assert status["active"] is True
    assert status["current_test"] == "static"
    assert status["session_setup_id"].startswith("session_")
    assert runtime.calibration.setup_metadata()["description"] == status["session_setup_id"]
    assert Path(status["run_directory"]).is_dir()
    cancelled = runtime.cancel_calibration_test()
    assert cancelled["active"] is False


class AutoDetectedCamera(ReconfigurableCamera):
    camera_index = "auto"
    active_camera_index = 3
    description = "camera:3 [V4L2, MJPG]"


def test_runtime_camera_status_reports_auto_selected_device(tmp_path: Path) -> None:
    runtime = VisionRuntime(
        AutoDetectedCamera(),
        FakeEstimator(),
        recording_root=tmp_path,
    )  # type: ignore[arg-type]

    camera = runtime.camera_status()

    assert camera["selector"] == "auto"
    assert camera["active_index"] == 3
    assert camera["source"] == "camera:3 [V4L2, MJPG]"


def test_runtime_reports_portable_clickable_export_paths(tmp_path: Path) -> None:
    recording_root = tmp_path / "video_recordings"
    calibration_root = tmp_path / "calibration_suggestions"
    runtime = VisionRuntime(
        PreviewCamera(),
        FakeEstimator(),
        recording_root=recording_root,
        calibration_root=calibration_root,
    )  # type: ignore[arg-type]

    status = runtime.status()

    assert status["recording"]["display_path"] == "video_recordings"
    assert status["recording"]["browse_url"] == "/exports/recordings/"
    assert status["calibration"]["display_path"] == "calibration_suggestions"
    assert status["calibration"]["browse_url"] == "/exports/calibration/"
    assert str(tmp_path) not in status["recording"]["display_path"]
    assert str(tmp_path) not in status["calibration"]["display_path"]


def test_pose_preset_is_versioned_atomic_and_loaded_on_restart(tmp_path: Path) -> None:
    recording_root = tmp_path / "video_recordings"
    calibration_root = tmp_path / "calibration_suggestions"
    runtime = VisionRuntime(
        PreviewCamera(), FakeEstimator(),
        recording_root=recording_root, calibration_root=calibration_root,
    )  # type: ignore[arg-type]
    values = {
        "confidence_threshold": 0.42,
        "landmark_smoothing": 0.61,
        "angle_smoothing": 0.27,
        "min_segment_length": 0.035,
        "max_angle_jump_deg": 52.0,
    }

    saved = runtime.save_tuning_preset(values)
    preset_path = Path(saved["path"])
    document = json.loads(preset_path.read_text(encoding="utf-8"))
    assert document == {"schema_version": 1, "pose_tuning": values}
    assert not list(calibration_root.glob("pose_tuning_preset_v1.json.tmp-*"))

    restarted = VisionRuntime(
        PreviewCamera(), FakeEstimator(),
        recording_root=recording_root, calibration_root=calibration_root,
    )  # type: ignore[arg-type]
    assert {key: restarted.tuning()[key] for key in values} == values
    assert restarted.pose_preset_status()["loaded"] is True
    assert restarted.pose_preset_status()["saved"] is True

    preset_path.write_text('{"schema_version":99,"pose_tuning":{}}', encoding="utf-8")
    rejected = VisionRuntime(
        PreviewCamera(), FakeEstimator(),
        recording_root=recording_root, calibration_root=calibration_root,
    )  # type: ignore[arg-type]
    assert "Could not load saved pose settings" in rejected.pose_preset_status()["error"]
    assert rejected.tuning()["confidence_threshold"] == 0.29


def test_pose_change_invalidates_calibration_and_is_blocked_during_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = VisionRuntime(
        PreviewCamera(), FakeEstimator(),
        recording_root=tmp_path / "video_recordings",
        calibration_root=tmp_path / "calibration_suggestions",
    )  # type: ignore[arg-type]
    invalidations: list[str] = []
    monkeypatch.setattr(runtime.calibration, "invalidate_setup", invalidations.append)

    runtime.update_tuning({"confidence_threshold": 0.41})
    assert invalidations and "Pose configuration changed" in invalidations[0]

    monkeypatch.setattr(runtime.recorder, "status", lambda: SimpleNamespace(active=True))
    with pytest.raises(RuntimeError, match="Stop recording"):
        runtime.update_tuning({"confidence_threshold": 0.43})
