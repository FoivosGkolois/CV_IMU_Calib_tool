from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from computer.vision_imu_calibration.recorder import AnnotatedVideoRecorder


class FakeWriter:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.frames: list[object] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV spelling
        return self.opened

    def write(self, frame: object) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    def __init__(self) -> None:
        self.writer = FakeWriter()
        self.calls: list[tuple[str, float, tuple[int, int]]] = []

    @staticmethod
    def VideoWriter_fourcc(*codec: str) -> int:  # noqa: N802
        return 1

    def VideoWriter(self, path: str, fourcc: int, fps: float, size: tuple[int, int]):  # noqa: N802
        self.calls.append((path, fps, size))
        return self.writer


def test_annotated_recorder_creates_video_metadata_and_timeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_cv2 = FakeCv2()
    monkeypatch.setattr("computer.vision_imu_calibration.recorder._load_cv2", lambda: fake_cv2)
    recorder = AnnotatedVideoRecorder(tmp_path)
    recorder.request_start(settings={"confidence_threshold": 0.55}, source="camera:0")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    recorder.write(
        frame,
        preferred_fps=12.5,
        sample={
            "frame_index": 1,
            "visual_knee_angle_deg": 42.0,
            "raw_knee_angle_deg": 43.0,
            "pose_confidence": 0.9,
            "tracking_valid": True,
            "tracking_reason": "ok",
            "tracked_leg": "right",
            "inference_fps": 12.5,
        },
    )
    status = recorder.stop()

    assert status.frame_count == 1
    assert fake_cv2.writer.released
    assert Path(status.csv_path).is_file()
    assert Path(status.metadata_path).is_file()
    metadata = json.loads(Path(status.metadata_path).read_text(encoding="utf-8"))
    assert metadata["source"] == "camera:0"
    assert metadata["record_fps"] == 12.5
    timeline = Path(status.csv_path).read_text(encoding="utf-8")
    assert "visual_knee_angle_deg" in timeline
    assert "42.0" in timeline


def test_recorder_resamples_video_to_monotonic_elapsed_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_cv2 = FakeCv2()
    monkeypatch.setattr("computer.vision_imu_calibration.recorder._load_cv2", lambda: fake_cv2)
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        "computer.vision_imu_calibration.recorder.time.perf_counter_ns",
        lambda: clock["now_ns"],
    )

    recorder = AnnotatedVideoRecorder(tmp_path)
    recorder.request_start(settings={}, source="camera:0")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    sample = {
        "frame_index": 1,
        "visual_knee_angle_deg": 10.0,
        "raw_knee_angle_deg": 10.0,
        "pose_confidence": 0.8,
        "tracking_valid": True,
        "tracking_reason": "ok",
        "tracked_leg": "right",
        "inference_fps": 10.0,
    }

    recorder.write(frame, preferred_fps=10.0, sample=sample)
    clock["now_ns"] = 11_000_000_000
    recorder.write(frame, preferred_fps=10.0, sample={**sample, "frame_index": 2})
    status = recorder.stop()

    assert status.frame_count == 2
    assert status.video_frame_count == 10
    assert len(fake_cv2.writer.frames) == 10
    metadata = json.loads(Path(status.metadata_path).read_text(encoding="utf-8"))
    assert metadata["timing_mode"] == "shared_pc_perf_counter_ns_nearest_sample"
    assert metadata["source_frame_count"] == 2
    assert metadata["video_frame_count"] == 10
    assert metadata["recording_elapsed_s"] == 1.0
    assert metadata["video_duration_s"] == 1.0


def test_recorder_writes_raw_imu_and_builds_synchronized_timeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from computer.vision_imu_calibration.imu_serial import ImuCalibrationSample

    fake_cv2 = FakeCv2()
    monkeypatch.setattr("computer.vision_imu_calibration.recorder._load_cv2", lambda: fake_cv2)
    clock = {"now_ns": 2_000_000_000}
    monkeypatch.setattr(
        "computer.vision_imu_calibration.recorder.time.perf_counter_ns",
        lambda: clock["now_ns"],
    )

    recorder = AnnotatedVideoRecorder(tmp_path)
    recorder.request_start(settings={}, source="camera:0", imu_source="COM3@921600")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    recorder.write(
        frame,
        preferred_fps=10.0,
        sample={
            "host_monotonic_ns": 2_010_000_000,
            "frame_index": 1,
            "visual_knee_angle_deg": 30.0,
            "raw_knee_angle_deg": 31.0,
            "pose_confidence": 0.9,
            "tracking_valid": True,
            "tracking_reason": "ok",
            "tracked_leg": "right",
            "inference_fps": 10.0,
            "hip_x": 0.4,
            "hip_y": 0.3,
            "knee_x": 0.5,
            "knee_y": 0.5,
            "ankle_x": 0.6,
            "ankle_y": 0.8,
        },
    )
    recorder.write_imu(
        ImuCalibrationSample(
            host_monotonic_ns=2_011_000_000,
            schema_version=1,
            sample_index=1,
            mcu_timestamp_us=10000,
            thigh_ok=True,
            shank_ok=True,
            thigh_ax_raw=1,
            thigh_ay_raw=2,
            thigh_az_raw=3,
            thigh_gx_raw=4,
            thigh_gy_raw=5,
            thigh_gz_raw=6,
            shank_ax_raw=7,
            shank_ay_raw=8,
            shank_az_raw=9,
            shank_gx_raw=10,
            shank_gy_raw=11,
            shank_gz_raw=12,
        )
    )
    clock["now_ns"] = 2_020_000_000
    status = recorder.stop()

    assert status.imu_sample_count == 1
    assert status.synchronized_row_count == 1
    assert Path(status.imu_csv_path).is_file()
    assert Path(status.synchronized_csv_path).is_file()
    synchronized = Path(status.synchronized_csv_path).read_text(encoding="utf-8")
    assert "sync_delta_ms" in synchronized
    assert "imu_sample_index" in synchronized
