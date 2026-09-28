from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .presentation import build_acquisition_presentation
from .synchronization import build_synchronized_timeline


@dataclass(frozen=True)
class RecordingStatus:
    active: bool
    path: str
    archive_path: str
    output_root: str
    metadata_path: str
    csv_path: str
    imu_csv_path: str
    estimator_csv_path: str
    synchronized_csv_path: str
    summary_path: str
    overview_plot_path: str
    elapsed_s: float
    frame_count: int
    video_frame_count: int
    imu_sample_count: int
    estimator_sample_count: int
    synchronized_row_count: int
    error: str


class AnnotatedVideoRecorder:
    """Record separate raw, processed, visual, and synchronized timelines."""

    REFERENCE_FIELDS = [
        "recording_time_s",
        "host_monotonic_ns",
        "frame_index",
        "visual_knee_angle_deg",
        "tracking_valid",
        "tracking_reason",
        "marker_quality",
        "thigh_tag_detected",
        "shank_tag_detected",
        "thigh_tag_center_x",
        "thigh_tag_center_y",
        "shank_tag_center_x",
        "shank_tag_center_y",
        "thigh_pose_valid",
        "shank_pose_valid",
        "thigh_reprojection_rmse_px",
        "shank_reprojection_rmse_px",
        "thigh_view_angle_deg",
        "shank_view_angle_deg",
        "processing_fps",
        "live_imu_filter_valid",
        "live_imu_knee_angle_deg",
        "live_imu_knee_speed_deg_s",
        "live_angle_difference_deg",
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

    ESTIMATOR_FIELDS = [
        "recording_time_s",
        "host_monotonic_ns",
        "sample_index",
        "mcu_timestamp_us",
        "filter_valid",
        "knee_angle_deg",
        "knee_speed_deg_s",
        "thigh_angle_deg",
        "shank_angle_deg",
        "filter_reason",
    ]

    def __init__(self, output_root: str | Path = "video_recordings") -> None:
        self.output_root = Path(output_root)
        self._lock = threading.RLock()
        self._requested = False
        self._writer: Any | None = None
        self._session_dir: Path | None = None
        self._session_name = ""
        self._archive_path: Path | None = None
        self._archive_finalized = False
        self._video_path: Path | None = None
        self._csv_path: Path | None = None
        self._imu_csv_path: Path | None = None
        self._estimator_csv_path: Path | None = None
        self._synchronized_csv_path: Path | None = None
        self._summary_path: Path | None = None
        self._overview_plot_path: Path | None = None
        self._metadata_path: Path | None = None
        self._csv_file: Any | None = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._imu_csv_file: Any | None = None
        self._imu_csv_writer: csv.DictWriter[str] | None = None
        self._estimator_csv_file: Any | None = None
        self._estimator_csv_writer: csv.DictWriter[str] | None = None
        self._started_monotonic_ns: int | None = None
        self._final_elapsed_s = 0.0
        self._frame_count = 0
        self._video_frame_count = 0
        self._imu_sample_count = 0
        self._estimator_sample_count = 0
        self._synchronized_row_count = 0
        self._last_frame: Any | None = None
        self._error = ""
        self._settings_snapshot: dict[str, Any] = {}
        self._source = ""
        self._imu_source = ""
        self._record_fps = 0.0
        self._metadata: dict[str, Any] = {}
        self._session_output_root: Path | None = None
        self._session_name_prefix = "vision_imu_session"
        self._exact_session_name: str | None = None
        self._extra_metadata: dict[str, Any] = {}

    def request_start(
        self,
        *,
        settings: Mapping[str, Any],
        source: str,
        imu_source: str = "",
        output_root: str | Path | None = None,
        session_name_prefix: str = "vision_imu_session",
        session_name: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if self._requested or self._writer is not None:
                return
            self._reset_session_state_locked()
            self._requested = True
            self._settings_snapshot = dict(settings)
            self._source = str(source)
            self._imu_source = str(imu_source)
            self._session_output_root = (
                Path(output_root) if output_root is not None else self.output_root
            )
            clean_prefix = "".join(
                character if character.isalnum() or character in {"-", "_"} else "_"
                for character in str(session_name_prefix or "vision_imu_session")
            ).strip("_")
            self._session_name_prefix = clean_prefix or "vision_imu_session"
            if session_name is None:
                self._exact_session_name = None
            else:
                clean_name = "".join(
                    character if character.isalnum() or character in {"-", "_"} else "_"
                    for character in str(session_name)
                ).strip("_")
                if not clean_name:
                    raise ValueError("session_name must contain at least one safe character")
                self._exact_session_name = clean_name
            self._extra_metadata = dict(extra_metadata or {})
            self._started_monotonic_ns = time.perf_counter_ns()
            self._create_session_locked()

    def stop(self) -> RecordingStatus:
        with self._lock:
            self._requested = False
            self._close_locked()
            return self._status_locked()

    def write(self, frame: Any, *, preferred_fps: float, sample: Mapping[str, Any]) -> None:
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
                    row = {field: sample.get(field) for field in self.REFERENCE_FIELDS}
                    row["recording_time_s"] = elapsed
                    row["host_monotonic_ns"] = host_ns
                    row["frame_index"] = sample.get("frame_index", self._frame_count)
                    self._csv_writer.writerow(row)
                    self._csv_file.flush()
            except Exception as exc:
                self._error = str(exc)
                self._requested = False
                self._close_locked()

    def write_imu(self, sample: Any, *, estimate: Any | None = None) -> None:
        with self._lock:
            if not self._requested or self._imu_csv_writer is None or self._started_monotonic_ns is None:
                return
            try:
                record = _to_record(sample)
                host_ns = int(record["host_monotonic_ns"])
                elapsed = max(0.0, (host_ns - self._started_monotonic_ns) / 1_000_000_000.0)
                record["recording_time_s"] = elapsed
                self._imu_csv_writer.writerow({field: record.get(field) for field in self.IMU_FIELDS})
                self._imu_csv_file.flush()
                self._imu_sample_count += 1

                if estimate is not None and self._estimator_csv_writer is not None:
                    estimator_record = _to_record(estimate)
                    estimator_record["recording_time_s"] = elapsed
                    estimator_record["host_monotonic_ns"] = host_ns
                    self._estimator_csv_writer.writerow(
                        {field: estimator_record.get(field) for field in self.ESTIMATOR_FIELDS}
                    )
                    self._estimator_csv_file.flush()
                    self._estimator_sample_count += 1
            except Exception as exc:
                self._error = str(exc)

    def status(self) -> RecordingStatus:
        with self._lock:
            return self._status_locked()

    def open_output_location(self) -> str:
        """Return the recorder output folder without invoking a platform-specific shell."""
        output_root = self.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        return str(output_root)

    def close(self) -> None:
        with self._lock:
            self._requested = False
            self._close_locked()

    def _reset_session_state_locked(self) -> None:
        self._error = ""
        self._session_dir = None
        self._session_name = ""
        self._archive_path = None
        self._archive_finalized = False
        self._video_path = None
        self._csv_path = None
        self._imu_csv_path = None
        self._estimator_csv_path = None
        self._synchronized_csv_path = None
        self._summary_path = None
        self._overview_plot_path = None
        self._metadata_path = None
        self._frame_count = 0
        self._video_frame_count = 0
        self._imu_sample_count = 0
        self._estimator_sample_count = 0
        self._synchronized_row_count = 0
        self._started_monotonic_ns = None
        self._final_elapsed_s = 0.0
        self._last_frame = None
        self._metadata = {}
        self._record_fps = 0.0
        self._session_output_root = None
        self._session_name_prefix = "vision_imu_session"
        self._exact_session_name = None
        self._extra_metadata = {}

    def _create_session_locked(self) -> None:
        output_root = self._session_output_root or self.output_root
        output_root.mkdir(parents=True, exist_ok=True)
        if self._exact_session_name is not None:
            session_name = self._exact_session_name
            archive_path = output_root / f"{session_name}.zip"
            if archive_path.exists():
                raise FileExistsError(
                    f"Deterministic recording output already exists: {archive_path}"
                )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"{self._session_name_prefix}_{timestamp}"
            archive_path = output_root / f"{session_name}.zip"
            suffix = 1
            while archive_path.exists():
                session_name = f"{self._session_name_prefix}_{timestamp}_{suffix:02d}"
                archive_path = output_root / f"{session_name}.zip"
                suffix += 1

        session_dir = Path(tempfile.mkdtemp(prefix=f"{session_name}_"))
        self._session_name = session_name
        self._archive_path = archive_path
        self._archive_finalized = False
        self._session_dir = session_dir
        self._csv_path = session_dir / "reference_timeline.csv"
        self._imu_csv_path = session_dir / "raw_imu.csv"
        self._estimator_csv_path = session_dir / "imu_estimator_timeline.csv"
        self._synchronized_csv_path = session_dir / "synchronized_timeline.csv"
        self._summary_path = session_dir / "acquisition_summary.json"
        self._overview_plot_path = session_dir / "acquisition_overview.png"
        self._metadata_path = session_dir / "recording_metadata.json"

        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.REFERENCE_FIELDS)
        self._csv_writer.writeheader()
        self._imu_csv_file = self._imu_csv_path.open("w", newline="", encoding="utf-8")
        self._imu_csv_writer = csv.DictWriter(self._imu_csv_file, fieldnames=self.IMU_FIELDS)
        self._imu_csv_writer.writeheader()
        self._estimator_csv_file = self._estimator_csv_path.open("w", newline="", encoding="utf-8")
        self._estimator_csv_writer = csv.DictWriter(
            self._estimator_csv_file, fieldnames=self.ESTIMATOR_FIELDS
        )
        self._estimator_csv_writer.writeheader()

        self._metadata = {
            "created_at": datetime.now().astimezone().isoformat(),
            "source": self._source,
            "imu_source": self._imu_source,
            "archive_file": self._archive_path.name,
            "archive_layout": "files_at_zip_root",
            "video_file": "",
            "timeline_file": self._csv_path.name,
            "raw_imu_file": self._imu_csv_path.name,
            "imu_estimator_file": self._estimator_csv_path.name,
            "synchronized_file": self._synchronized_csv_path.name,
            "acquisition_summary_file": self._summary_path.name,
            "acquisition_overview_file": self._overview_plot_path.name,
            "timing_mode": "shared_pc_perf_counter_ns_nearest_sample",
            "session_start_host_monotonic_ns": self._started_monotonic_ns,
            "tuning": self._settings_snapshot,
            **self._extra_metadata,
        }
        self._write_metadata_locked()

    def _open_video_locked(self, frame: Any, preferred_fps: float) -> None:
        if self._session_dir is None:
            raise RuntimeError("Recording session directory is not initialized")
        cv2 = _load_cv2()
        height, width = frame.shape[:2]
        fps = min(30.0, max(5.0, round(float(preferred_fps or 15.0), 1)))
        mp4_path = self._session_dir / "annotated_reference.mp4"
        writer = cv2.VideoWriter(
            str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        if not writer.isOpened():
            writer.release()
            avi_path = self._session_dir / "annotated_reference.avi"
            writer = cv2.VideoWriter(
                str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (int(width), int(height))
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
        if self._archive_finalized:
            return
        if self._started_monotonic_ns is not None:
            self._final_elapsed_s = max(
                0.0, (time.perf_counter_ns() - self._started_monotonic_ns) / 1_000_000_000.0
            )
            if self._last_frame is not None:
                self._write_video_until_locked(self._final_elapsed_s, self._last_frame)

        writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()
        for file_attr in ("_csv_file", "_imu_csv_file", "_estimator_csv_file"):
            file_obj = getattr(self, file_attr)
            setattr(self, file_attr, None)
            if file_obj is not None:
                file_obj.flush()
                file_obj.close()
        self._csv_writer = None
        self._imu_csv_writer = None
        self._estimator_csv_writer = None

        if self._csv_path and self._imu_csv_path and self._synchronized_csv_path:
            try:
                self._synchronized_row_count = build_synchronized_timeline(
                    self._csv_path,
                    self._imu_csv_path,
                    self._synchronized_csv_path,
                    estimator_csv_path=self._estimator_csv_path,
                )
                if self._summary_path and self._overview_plot_path:
                    build_acquisition_presentation(
                        self._synchronized_csv_path,
                        self._summary_path,
                        self._overview_plot_path,
                    )
            except Exception as exc:
                self._error = self._error or f"Acquisition presentation failed: {exc}"

        if self._metadata_path is not None and self._metadata:
            self._metadata.update(
                {
                    "source_frame_count": self._frame_count,
                    "video_frame_count": self._video_frame_count,
                    "raw_imu_sample_count": self._imu_sample_count,
                    "imu_estimator_sample_count": self._estimator_sample_count,
                    "synchronized_row_count": self._synchronized_row_count,
                    "recording_elapsed_s": self._final_elapsed_s,
                    "video_duration_s": (
                        self._video_frame_count / self._record_fps if self._record_fps > 0.0 else 0.0
                    ),
                }
            )
            self._write_metadata_locked()

        try:
            self._finalize_archive_locked()
        except Exception as exc:
            staging = str(self._session_dir or "")
            self._error = self._error or f"ZIP finalization failed: {exc}. Recovery files: {staging}"

    def _finalize_archive_locked(self) -> None:
        if self._archive_finalized or self._session_dir is None or self._archive_path is None:
            return
        if not self._session_dir.is_dir():
            raise RuntimeError(f"Recording staging directory is missing: {self._session_dir}")

        archive_root = self._archive_path.parent
        archive_root.mkdir(parents=True, exist_ok=True)
        temporary_archive = self._archive_path.with_suffix(".zip.partial")
        temporary_archive.unlink(missing_ok=True)
        try:
            with zipfile.ZipFile(temporary_archive, "w") as archive:
                for source in sorted(self._session_dir.rglob("*")):
                    if not source.is_file():
                        continue
                    compression = (
                        zipfile.ZIP_STORED
                        if source.suffix.lower() in {".mp4", ".avi"}
                        else zipfile.ZIP_DEFLATED
                    )
                    archive.write(
                        source,
                        arcname=source.relative_to(self._session_dir).as_posix(),
                        compress_type=compression,
                    )
            os.replace(temporary_archive, self._archive_path)
        finally:
            temporary_archive.unlink(missing_ok=True)

        shutil.rmtree(self._session_dir)
        self._archive_finalized = True

    def _write_metadata_locked(self) -> None:
        if self._metadata_path is not None:
            self._metadata_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")

    def _status_locked(self) -> RecordingStatus:
        active = self._requested or self._writer is not None
        elapsed = (
            max(0.0, (time.perf_counter_ns() - self._started_monotonic_ns) / 1_000_000_000.0)
            if self._started_monotonic_ns is not None and active
            else self._final_elapsed_s
        )
        archive_path = str(self._archive_path.resolve()) if self._archive_path is not None else ""
        active_output_root = self._session_output_root or self.output_root
        output_root = str(active_output_root.expanduser().resolve())
        return RecordingStatus(
            active=active,
            path=archive_path,
            archive_path=archive_path,
            output_root=output_root,
            metadata_path=self._reported_member_path(self._metadata_path),
            csv_path=self._reported_member_path(self._csv_path),
            imu_csv_path=self._reported_member_path(self._imu_csv_path),
            estimator_csv_path=self._reported_member_path(self._estimator_csv_path),
            synchronized_csv_path=self._reported_member_path(self._synchronized_csv_path),
            summary_path=self._reported_member_path(self._summary_path),
            overview_plot_path=self._reported_member_path(self._overview_plot_path),
            elapsed_s=elapsed,
            frame_count=self._frame_count,
            video_frame_count=self._video_frame_count,
            imu_sample_count=self._imu_sample_count,
            estimator_sample_count=self._estimator_sample_count,
            synchronized_row_count=self._synchronized_row_count,
            error=self._error,
        )

    def _reported_member_path(self, path: Path | None) -> str:
        if path is None:
            return ""
        if self._archive_finalized and self._archive_path is not None:
            return f"{self._archive_path.resolve()}::{path.name}"
        return str(path)


def _to_record(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_record"):
        return dict(value.to_record())
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed") from exc
    return cv2
