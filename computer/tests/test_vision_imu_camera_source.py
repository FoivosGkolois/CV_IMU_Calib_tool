from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from computer.vision_imu_calibration.camera_source import OpenCvCameraSource


class FakeCapture:
    def __init__(self, *, opened: bool, frames: list[object | None]) -> None:
        self.opened = opened
        self.frames = list(frames)
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API spelling
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        frame = self.frames.pop(0) if self.frames else None
        return frame is not None, frame

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        return True

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_MSMF = 1400
    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, captures: dict[tuple[int, int | None], FakeCapture]) -> None:
        self.captures = captures
        self.calls: list[tuple[int, int | None]] = []

    def VideoCapture(self, source: int, backend: int | None = None) -> FakeCapture:  # noqa: N802
        key = (source, backend)
        self.calls.append(key)
        return self.captures[key]

    @staticmethod
    def flip(frame: object, axis: int) -> object:
        return ("flipped", frame, axis)


def test_camera_source_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        OpenCvCameraSource(camera_index=None, video_path=None)
    with pytest.raises(ValueError):
        OpenCvCameraSource(camera_index=0, video_path=Path("video.mp4"))


def test_camera_description_is_explicit() -> None:
    assert OpenCvCameraSource(camera_index=2).description == "camera:2"


def test_windows_auto_prefers_dshow_for_fast_startup() -> None:
    dshow = FakeCapture(opened=True, frames=["first-frame"])
    msmf = FakeCapture(opened=False, frames=[])
    default = FakeCapture(opened=False, frames=[])
    cv2 = FakeCv2({(0, 700): dshow, (0, 1400): msmf, (0, None): default})
    source = OpenCvCameraSource(camera_index=0, cv2_module=cv2, platform_name="Windows")

    source.open()

    assert cv2.calls == [(0, 700)]
    assert source.description == "camera:0 [DSHOW]"
    assert source.read() == ("flipped", "first-frame", 1)


def test_windows_camera_falls_back_when_dshow_returns_no_frames() -> None:
    dshow = FakeCapture(opened=True, frames=[None, None, None, None, None, None])
    msmf = FakeCapture(opened=True, frames=["usable-frame"])
    default = FakeCapture(opened=False, frames=[])
    cv2 = FakeCv2({(0, 700): dshow, (0, 1400): msmf, (0, None): default})
    source = OpenCvCameraSource(camera_index=0, cv2_module=cv2, platform_name="Windows")

    source.open()

    assert dshow.released is True
    assert source.description == "camera:0 [MSMF]"
    assert source.read() == ("flipped", "usable-frame", 1)


def test_camera_open_error_lists_attempted_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("computer.vision_imu_calibration.camera_source.time.sleep", lambda _: None)
    captures = {
        (0, 1400): FakeCapture(opened=False, frames=[]),
        (0, 700): FakeCapture(opened=False, frames=[]),
        (0, None): FakeCapture(opened=False, frames=[]),
    }
    source = OpenCvCameraSource(
        camera_index=0,
        cv2_module=FakeCv2(captures),
        platform_name="Windows",
    )

    with pytest.raises(RuntimeError, match="DSHOW: open failed; MSMF: open failed; DEFAULT: open failed"):
        source.open()


def test_explicit_windows_backend_avoids_fallback_probe() -> None:
    msmf = FakeCapture(opened=True, frames=["msmf-frame"])
    cv2 = FakeCv2({(0, 1400): msmf})
    source = OpenCvCameraSource(
        camera_index=0,
        backend="msmf",
        cv2_module=cv2,
        platform_name="Windows",
    )

    source.open()

    assert cv2.calls == [(0, 1400)]
    assert source.description == "camera:0 [MSMF]"


def test_camera_backend_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="Unsupported camera backend"):
        OpenCvCameraSource(camera_index=0, backend="mystery")


def test_default_camera_profile_is_lightweight() -> None:
    source = OpenCvCameraSource(camera_index=0)
    assert source.width == 640
    assert source.height == 480
    assert source.fps == 30.0
