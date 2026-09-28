from __future__ import annotations

import threading
import time
from pathlib import Path

from computer.vision_imu_calibration.marker_reference import MarkerReferenceResult
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


class FakeReference:
    description = "fake-apriltag-reference"
    thigh_tag_id = 0
    shank_tag_id = 1

    def __init__(self) -> None:
        self.closed = False

    def open(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def process(self, frame):
        return MarkerReferenceResult(
            angle_deg=None,
            valid=False,
            reason="reference_setup_required",
            quality=1.0,
            thigh=None,
            shank=None,
        )


def test_runtime_start_returns_before_slow_camera_initialization(tmp_path: Path) -> None:
    camera = BlockingCamera()
    runtime = VisionRuntime(camera, FakeReference(), recording_root=tmp_path)  # type: ignore[arg-type]
    started = time.perf_counter()
    runtime.start()
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2
    assert runtime.status()["state"] == "STARTING"
    camera.release_open.set()
    runtime.stop()


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


class FailingReference(FakeReference):
    def open(self) -> None:
        raise RuntimeError("tag backend unavailable")


def test_runtime_publishes_camera_preview_when_reference_backend_fails(tmp_path: Path) -> None:
    camera = PreviewCamera()
    runtime = VisionRuntime(camera, FailingReference(), recording_root=tmp_path)  # type: ignore[arg-type]
    runtime.start()

    sequence, jpeg = runtime.wait_for_jpeg(0, timeout_s=1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and runtime.status()["state"] != "REFERENCE_FAILED":
        time.sleep(0.01)
    status = runtime.status()

    assert sequence >= 1
    assert jpeg is not None
    assert status["state"] == "REFERENCE_FAILED"
    assert status["frame_index"] >= 1
    assert status["tracking_reason"] == "reference_backend_failed"
    assert "tag backend unavailable" in status["error"]
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
        FakeReference(),
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
    runtime = VisionRuntime(camera, FakeReference(), recording_root=tmp_path)  # type: ignore[arg-type]
    runtime._status = RuntimeStatus(
        state="RUNNING",
        source=camera.description,
        tracking_valid=False,
        tracking_reason="test",
        visual_knee_angle_deg=None,
        marker_quality=0.0,
        thigh_tag_detected=False,
        shank_tag_detected=False,
        processing_fps=0.0,
        frame_index=0,
        error="",
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
        FakeReference(),
        recording_root=tmp_path,
        imu_source_factory=StreamingManualImuSource,
    )  # type: ignore[arg-type]
    status = runtime.connect_imu("COM8")
    assert status["capture_active"] is True
    runtime.disconnect_imu()


def test_acquisition_history_is_bounded_and_decimated(tmp_path: Path) -> None:
    runtime = VisionRuntime(PreviewCamera(), FakeReference(), recording_root=tmp_path)  # type: ignore[arg-type]
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
