from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .synchronization import build_synchronized_timeline


@dataclass(frozen=True)
class RecordingStatus:
    active: bool
    path: str
    metadata_path: str
    csv_path: str
    imu_csv_path: str
    synchronized_csv_path: str
    elapsed_s: float
    frame_count: int
    video_frame_count: int
    imu_sample_count: int
    synchronized_row_count: int
    error: str


class AnnotatedVideoRecorder:
    """Record a common host-time camera and raw-IMU calibration session.

    Camera and serial samples use ``time.perf_counter_ns()`` from the same PC.
    The raw timelines remain the source of truth. On close, each camera sample is
    paired with the nearest raw IMU sample into ``synchronized_timeline.csv``.
    """

    POSE_FIELDS = [
        "recording_time_s",
        "host_monotonic_ns",
        "frame_index",
        "visual_knee_angle_deg",
        "raw_knee_angle_deg",
        "pose_confidence",
        "tracking_valid",
        "tracking_reason",
        "tracked_leg",
        "inference_fps",
        "hip_x",
        "hip_y",
        "knee_x",
        "knee_y",
        "ankle_x",
        "ankle_y",
    ]

    IMU_FIELDS = [
        "recording_time_s",
        "host_monotonic_ns",
        "schema_version",
        "sample_index",
        "mcu_timestamp_us",
        "thigh_ok",
        "shank_ok",
        "thigh_ax_raw",
        "thigh_ay_raw",
        "thigh_az_raw",
        "thigh_gx_raw",
        "thigh_gy_raw",
        "thigh_gz_raw",
        "shank_ax_raw",
        "shank_ay_raw",
        "shank_az_raw",
        "shank_gx_raw",
        "shank_gy_raw",
        "shank_gz_raw",
        "thigh_ax_g",
        "thigh_ay_g",
        "thigh_az_g",
        "thigh_gx_dps",
        "thigh_gy_dps",
        "thigh_gz_dps",
        "shank_ax_g",
        "shank_ay_g",
        "shank_az_g",
        "shank_gx_dps",
        "shank_gy_dps",
        "shank_gz_dps",
    ]

    def __init__(self, output_root: str | Path = "vision_calibration_recordings") -> None:
        self.output_root = Path(output_root)
        self._lock = threading.RLock()
        self._requested = False
        self._writer: Any | None = None
        self._session_dir: Path | None = None
        self._video_path: Path | None = None
        self._csv_path: Path | None = None
        self._imu_csv_path: Path | None = None
        self._synchronized_csv_path: Path | None = None
        self._metadata_path: Path | None = None
        self._csv_file: Any | None = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._imu_csv_file: Any | None = None
        self._imu_csv_writer: csv.DictWriter[str] | None = None
        self._started_monotonic_ns: int | None = None
        self._final_elapsed_s = 0.0
        self._frame_count = 0
        self._video_frame_count = 0
        self._imu_sample_count = 0
        self._synchronized_row_count = 0
        self._last_frame: Any | None = None
        self._error = ""
        self._settings_snapshot: dict[str, Any] = {}
        self._source = ""
        self._imu_source = ""
        self._record_fps = 0.0
        self._metadata: dict[str, Any] = {}

    def request_start(
        self,
        *,
        settings: Mapping[str, Any],
        source: str,
        imu_source: str = "",
    ) -> None:
        with self._lock:
            if self._requested or self._writer is not None:
                return
            self._reset_session_state_locked()
            self._requested = True
            self._settings_snapshot = dict(settings)
            self._source = str(source)
            self._imu_source = str(imu_source)
            self._started_monotonic_ns = time.perf_counter_ns()
            self._create_session_locked()

    def stop(self) -> RecordingStatus:
        with self._lock:
            self._requested = False
            self._close_locked()
            return self._status_locked()

    def write(
        self,
        frame: Any,
        *,
        preferred_fps: float,
        sample: Mapping[str, Any],
    ) -> None:
        with self._lock:
            if not self._requested and self._writer is None:
                return
            try:
                if self._session_dir is None:
                    return
                if self._writer is None:
                    self._open_video_locked(frame, preferred_fps)
                if self._writer is None or self._started_monotonic_ns is None:
                    return

                host_ns = int(sample.get("host_monotonic_ns") or time.perf_counter_ns())
                elapsed = max(0.0, (host_ns - self._started_monotonic_ns) / 1_000_000_000.0)
                self._last_frame = frame.copy() if hasattr(frame, "copy") else frame
                self._write_video_until_locked(elapsed, self._last_frame)
                self._frame_count += 1

                if self._csv_writer is not None:
                    row = {field: sample.get(field) for field in self.POSE_FIELDS}
                    row["recording_time_s"] = elapsed
                    row["host_monotonic_ns"] = host_ns
                    row["frame_index"] = sample.get("frame_index", self._frame_count)
                    self._csv_writer.writerow(row)
                    self._csv_file.flush()
            except Exception as exc:
                self._error = str(exc)
                self._requested = False
                self._close_locked()

    def write_imu(self, sample: Any) -> None:
        with self._lock:
            if not self._requested or self._imu_csv_writer is None or self._started_monotonic_ns is None:
                return
            try:
                if hasattr(sample, "to_record"):
                    record = dict(sample.to_record())
                elif is_dataclass(sample):
                    record = asdict(sample)
                else:
                    record = dict(sample)
                host_ns = int(record["host_monotonic_ns"])
                record["recording_time_s"] = max(
                    0.0,
                    (host_ns - self._started_monotonic_ns) / 1_000_000_000.0,
                )
                row = {field: record.get(field) for field in self.IMU_FIELDS}
                self._imu_csv_writer.writerow(row)
                self._imu_csv_file.flush()
                self._imu_sample_count += 1
            except Exception as exc:
                self._error = str(exc)

    def status(self) -> RecordingStatus:
        with self._lock:
            return self._status_locked()

    def close(self) -> None:
        with self._lock:
            self._requested = False
            self._close_locked()

    def _reset_session_state_locked(self) -> None:
        self._error = ""
        self._session_dir = None
        self._video_path = None
        self._csv_path = None
        self._imu_csv_path = None
        self._synchronized_csv_path = None
        self._metadata_path = None
        self._frame_count = 0
        self._video_frame_count = 0
        self._imu_sample_count = 0
        self._synchronized_row_count = 0
        self._started_monotonic_ns = None
        self._final_elapsed_s = 0.0
        self._last_frame = None
        self._metadata = {}
        self._record_fps = 0.0

    def _create_session_locked(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_root / f"vision_imu_session_{timestamp}"
        suffix = 1
        while session_dir.exists():
            session_dir = self.output_root / f"vision_imu_session_{timestamp}_{suffix:02d}"
            suffix += 1
        session_dir.mkdir(parents=True, exist_ok=False)
        self._session_dir = session_dir
        self._csv_path = session_dir / "pose_timeline.csv"
        self._imu_csv_path = session_dir / "raw_imu.csv"
        self._synchronized_csv_path = session_dir / "synchronized_timeline.csv"
        self._metadata_path = session_dir / "recording_metadata.json"

        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.POSE_FIELDS)
        self._csv_writer.writeheader()
        self._imu_csv_file = self._imu_csv_path.open("w", newline="", encoding="utf-8")
        self._imu_csv_writer = csv.DictWriter(self._imu_csv_file, fieldnames=self.IMU_FIELDS)
        self._imu_csv_writer.writeheader()

        self._metadata = {
            "created_at": datetime.now().astimezone().isoformat(),
            "source": self._source,
            "imu_source": self._imu_source,
            "video_file": "",
            "timeline_file": self._csv_path.name,
            "raw_imu_file": self._imu_csv_path.name,
            "synchronized_file": self._synchronized_csv_path.name,
            "timing_mode": "shared_pc_perf_counter_ns_nearest_sample",
            "session_start_host_monotonic_ns": self._started_monotonic_ns,
            "tuning": self._settings_snapshot,
        }
        self._write_metadata_locked()

    def _open_video_locked(self, frame: Any, preferred_fps: float) -> None:
        if self._session_dir is None:
            raise RuntimeError("Recording session directory is not initialized")
        cv2 = _load_cv2()
        height, width = frame.shape[:2]
        fps = min(30.0, max(5.0, round(float(preferred_fps or 15.0), 1)))

        mp4_path = self._session_dir / "annotated_pose.mp4"
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            writer.release()
            avi_path = self._session_dir / "annotated_pose.avi"
            writer = cv2.VideoWriter(
                str(avi_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (int(width), int(height)),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("Could not open MP4 or AVI video writer")
            self._video_path = avi_path
        else:
            self._video_path = mp4_path

        self._writer = writer
        self._record_fps = fps
        self._metadata.update(
            {
                "video_file": self._video_path.name,
                "record_fps": fps,
                "frame_width": int(width),
                "frame_height": int(height),
            }
        )
        self._write_metadata_locked()

    def _write_video_until_locked(self, elapsed_s: float, frame: Any) -> None:
        if self._writer is None or self._record_fps <= 0.0:
            return
        target_count = max(1, int(round(float(elapsed_s) * self._record_fps)))
        while self._video_frame_count < target_count:
            self._writer.write(frame)
            self._video_frame_count += 1

    def _close_locked(self) -> None:
        if self._started_monotonic_ns is not None:
            self._final_elapsed_s = max(
                0.0,
                (time.perf_counter_ns() - self._started_monotonic_ns) / 1_000_000_000.0,
            )
            if self._last_frame is not None:
                self._write_video_until_locked(self._final_elapsed_s, self._last_frame)

        writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()
        csv_file, self._csv_file = self._csv_file, None
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
        imu_file, self._imu_csv_file = self._imu_csv_file, None
        if imu_file is not None:
            imu_file.flush()
            imu_file.close()
        self._csv_writer = None
        self._imu_csv_writer = None

        if (
            self._csv_path is not None
            and self._imu_csv_path is not None
            and self._synchronized_csv_path is not None
        ):
            try:
                self._synchronized_row_count = build_synchronized_timeline(
                    self._csv_path,
                    self._imu_csv_path,
                    self._synchronized_csv_path,
                )
            except Exception as exc:
                self._error = self._error or f"Synchronization failed: {exc}"

        if self._metadata_path is not None and self._metadata:
            self._metadata.update(
                {
                    "source_frame_count": self._frame_count,
                    "video_frame_count": self._video_frame_count,
                    "raw_imu_sample_count": self._imu_sample_count,
                    "synchronized_row_count": self._synchronized_row_count,
                    "recording_elapsed_s": self._final_elapsed_s,
                    "video_duration_s": (
                        self._video_frame_count / self._record_fps if self._record_fps > 0.0 else 0.0
                    ),
                }
            )
            self._write_metadata_locked()

    def _write_metadata_locked(self) -> None:
        if self._metadata_path is not None:
            self._metadata_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")

    def _status_locked(self) -> RecordingStatus:
        active = self._requested or self._writer is not None
        if self._started_monotonic_ns is not None and active:
            elapsed = max(
                0.0,
                (time.perf_counter_ns() - self._started_monotonic_ns) / 1_000_000_000.0,
            )
        else:
            elapsed = self._final_elapsed_s
        return RecordingStatus(
            active=active,
            path=str(self._video_path or self._session_dir or ""),
            metadata_path=str(self._metadata_path or ""),
            csv_path=str(self._csv_path or ""),
            imu_csv_path=str(self._imu_csv_path or ""),
            synchronized_csv_path=str(self._synchronized_csv_path or ""),
            elapsed_s=elapsed,
            frame_count=self._frame_count,
            video_frame_count=self._video_frame_count,
            imu_sample_count=self._imu_sample_count,
            synchronized_row_count=self._synchronized_row_count,
            error=self._error,
        )


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed") from exc
    return cv2
