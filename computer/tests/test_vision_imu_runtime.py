from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from computer.vision_imu_calibration.geometry import KneeAngleResult
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
