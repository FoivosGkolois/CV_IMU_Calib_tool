from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

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


def _zip_text(archive_path: str, member: str) -> str:
    with ZipFile(archive_path) as archive:
        return archive.read(member).decode("utf-8")


def _zip_json(archive_path: str, member: str) -> dict:
    return json.loads(_zip_text(archive_path, member))


def test_annotated_recorder_finalizes_one_zip_without_loose_trial_folder(
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
            "visual_knee_angle_deg": 142.0,
            "raw_knee_angle_deg": 143.0,
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
    assert Path(status.archive_path).is_file()
    assert list(tmp_path.glob("*.zip")) == [Path(status.archive_path)]
    assert not any(path.is_dir() for path in tmp_path.iterdir())
    with ZipFile(status.archive_path) as archive:
        names = set(archive.namelist())
    assert "recording_metadata.json" in names
    assert "pose_timeline.csv" in names
    assert all("/" not in name.rstrip("/") for name in names)
    metadata = _zip_json(status.archive_path, "recording_metadata.json")
    assert metadata["source"] == "camera:0"
    assert metadata["record_fps"] == 12.5
    assert metadata["archive_layout"] == "files_at_zip_root"
    timeline = _zip_text(status.archive_path, "pose_timeline.csv")
    assert "visual_knee_angle_deg" in timeline
    assert "142.0" in timeline


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
        "visual_knee_angle_deg": 170.0,
        "raw_knee_angle_deg": 170.0,
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
    metadata = _zip_json(status.archive_path, "recording_metadata.json")
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
            "visual_knee_angle_deg": 150.0,
            "raw_knee_angle_deg": 149.0,
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
    raw_imu = _zip_text(status.archive_path, "raw_imu.csv")
    synchronized = _zip_text(status.archive_path, "synchronized_timeline.csv")
    assert "thigh_ax_raw" in raw_imu
    assert "sync_delta_ms" in synchronized
    assert "imu_sample_index" in synchronized


def test_recorder_separates_raw_and_processed_imu_outputs(tmp_path: Path, monkeypatch) -> None:
    from computer.vision_imu_calibration.imu_filter import KneeAngleEstimate
    from computer.vision_imu_calibration.imu_serial import ImuCalibrationSample

    fake_cv2 = FakeCv2()
    monkeypatch.setattr("computer.vision_imu_calibration.recorder._load_cv2", lambda: fake_cv2)
    clock = {"now_ns": 3_000_000_000}
    monkeypatch.setattr(
        "computer.vision_imu_calibration.recorder.time.perf_counter_ns",
        lambda: clock["now_ns"],
    )
    recorder = AnnotatedVideoRecorder(tmp_path)
    recorder.request_start(settings={"imu_filter": {"implementation_id": "test"}}, source="camera:0")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    recorder.write(
        frame,
        preferred_fps=10.0,
        sample={
            "host_monotonic_ns": 3_010_000_000,
            "frame_index": 1,
            "visual_knee_angle_deg": 155.0,
            "tracking_valid": True,
        },
    )
    raw = ImuCalibrationSample(
        host_monotonic_ns=3_011_000_000,
        schema_version=1,
        sample_index=4,
        mcu_timestamp_us=40_000,
        thigh_ok=True,
        shank_ok=True,
        thigh_ax_raw=0,
        thigh_ay_raw=0,
        thigh_az_raw=16384,
        thigh_gx_raw=0,
        thigh_gy_raw=0,
        thigh_gz_raw=0,
        shank_ax_raw=0,
        shank_ay_raw=-8000,
        shank_az_raw=14000,
        shank_gx_raw=0,
        shank_gy_raw=0,
        shank_gz_raw=0,
    )
    estimate = KneeAngleEstimate(
        host_monotonic_ns=3_011_000_000,
        sample_index=4,
        mcu_timestamp_us=40_000,
        valid=True,
        knee_angle_deg=156.0,
        knee_speed_deg_s=-2.0,
        thigh_angle_deg=0.0,
        shank_angle_deg=24.0,
        reason="ok",
    )
    recorder.write_imu(raw, estimate=estimate)
    clock["now_ns"] = 3_020_000_000
    status = recorder.stop()
    assert status.estimator_sample_count == 1
    estimator_text = _zip_text(status.archive_path, "imu_estimator_timeline.csv")
    assert "knee_angle_deg" in estimator_text
    synchronized = _zip_text(status.archive_path, "synchronized_timeline.csv")
    assert "estimator_knee_angle_deg" in synchronized
    with ZipFile(status.archive_path) as archive:
        assert "acquisition_summary.json" in archive.namelist()
        assert "acquisition_overview.png" in archive.namelist()


def test_open_output_location_returns_portable_directory_without_shell_side_effects(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "video_recordings"
    recorder = AnnotatedVideoRecorder(output_root)

    resolved = recorder.open_output_location()

    assert Path(resolved) == output_root.resolve()
    assert Path(resolved).is_dir()


def test_recorder_supports_exact_owned_calibration_archive_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_cv2 = FakeCv2()
    monkeypatch.setattr("computer.vision_imu_calibration.recorder._load_cv2", lambda: fake_cv2)
    recorder = AnnotatedVideoRecorder(tmp_path / "video_recordings")
    calibration_root = tmp_path / "calibration_suggestions" / "calibration_run_test"
    recorder.request_start(
        settings={},
        source="camera:0",
        output_root=calibration_root,
        session_name="static_trial",
        extra_metadata={"calibration_test_type": "static"},
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    recorder.write(
        frame,
        preferred_fps=10.0,
        sample={
            "frame_index": 1,
            "visual_knee_angle_deg": 150.0,
            "tracking_valid": True,
        },
    )
    status = recorder.stop()

    archive = Path(status.archive_path)
    assert archive.parent == calibration_root.resolve()
    assert archive.name == "static_trial.zip"
    assert status.output_root == str(calibration_root.resolve())
    metadata = _zip_json(status.archive_path, "recording_metadata.json")
    assert metadata["calibration_test_type"] == "static"


def test_exact_session_name_refuses_to_overwrite_existing_archive(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "calibration_suggestions" / "calibration_run_test"
    output_root.mkdir(parents=True)
    (output_root / "static_trial.zip").write_bytes(b"existing")
    recorder = AnnotatedVideoRecorder(tmp_path / "video_recordings")

    with pytest.raises(FileExistsError):
        recorder.request_start(
            settings={},
            source="camera:0",
            output_root=output_root,
            session_name="static_trial",
        )
