from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .calibration_workflow import CalibrationWorkflow
from .camera_source import OpenCvCameraSource
from .geometry import KneeAngleResult, calculate_knee_flexion_deg
from .imu_filter import (
    KneeAngleEstimate,
    KneeAngleFilterMirror,
    KneeAngleFilterParameters,
)
from .imu_serial import IMUCAL_BAUD_RATE, SerialImuSource, list_serial_ports
from .landmark_filter import LegLandmarkEmaFilter
from .models import LegSide
from .overlay import draw_leg_overlay
from .pose_estimator import PoseEstimator
from .recorder import AnnotatedVideoRecorder
from .tuning import FrameRotation, RuntimeTuning


CAMERA_PROFILES: dict[str, tuple[int, int, float]] = {
    "fast": (640, 480, 30.0),
    "quality": (1280, 720, 30.0),
}


def _empty_imu_status() -> dict[str, Any]:
    return {
        "configured": False,
        "connected": False,
        "capture_active": False,
        "port": "",
        "baud_rate": 0,
        "transport": "none",
        "state": "DISCONNECTED",
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


def _optional_owned_path(
    value: Any,
    root: Path,
    *,
    directory: bool = False,
    require_exists: bool = True,
) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text.split("::", 1)[0]).expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not require_exists:
        return candidate
    if directory:
        return candidate if candidate.is_dir() else None
    return candidate if candidate.is_file() else None


@dataclass(frozen=True)
class RuntimeStatus:
    state: str
    source: str
    requested_leg: str
    tracked_leg: str | None
    tracking_valid: bool
    tracking_reason: str
    visual_knee_angle_deg: float | None
    raw_knee_angle_deg: float | None
    pose_confidence: float
    inference_fps: float
    frame_index: int
    error: str


class VisionRuntime:
    """Own camera, pose inference, live tuning, annotated stream, and recorder.

    Camera capture starts independently from pose-model initialization. This keeps
    the live browser preview available while the pose model loads and also preserves a
    camera-only diagnostic preview if pose initialization fails.
    """

    def __init__(
        self,
        camera: OpenCvCameraSource,
        estimator: PoseEstimator,
        *,
        leg_side: LegSide = LegSide.RIGHT,
        confidence_threshold: float = 0.30,
        recording_root: str | Path = "video_recordings",
        calibration_root: str | Path = "calibration_suggestions",
        imu_source: Any | None = None,
        imu_source_factory: Any = SerialImuSource,
        imu_filter: KneeAngleFilterMirror | None = None,
        imu_filter_config_path: str | Path | None = None,
    ) -> None:
        self.camera = camera
        self.estimator = estimator
        self._tuning = RuntimeTuning(
            leg_side=leg_side,
            confidence_threshold=float(confidence_threshold),
        )
        self.recording_root = Path(recording_root).expanduser().resolve()
        self.calibration_root = Path(calibration_root).expanduser().resolve()
        self.recording_root.mkdir(parents=True, exist_ok=True)
        self.calibration_root.mkdir(parents=True, exist_ok=True)
        self.recorder = AnnotatedVideoRecorder(self.recording_root)
        self.calibration = CalibrationWorkflow(self.calibration_root)
        self._calibration_timer_cancel: threading.Event | None = None
        self._calibration_timer_thread: threading.Thread | None = None
        self._calibration_action_lock = threading.RLock()
        self._imu_source = imu_source
        self._imu_source_factory = imu_source_factory
        self._imu_lock = threading.RLock()
        self._imu_filter_lock = threading.RLock()
        self._imu_filter_config_path = (
            Path(imu_filter_config_path).resolve() if imu_filter_config_path is not None else None
        )
        self._imu_filter = imu_filter or KneeAngleFilterMirror(KneeAngleFilterParameters())
        self._latest_imu_estimate = KneeAngleEstimate()
        self._acquisition_history: deque[dict[str, Any]] = deque(maxlen=1200)
        self._lock = threading.RLock()
        self._new_frame = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._pose_ready = threading.Event()
        self._pose_failed = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._imu_thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_sequence = 0
        self._pose_error = ""
        self._reset_tracking_event = threading.Event()
        self._camera_profile = self._infer_camera_profile()
        self._status = RuntimeStatus(
            state="IDLE",
            source=self.camera.description,
            requested_leg=self._tuning.leg_side.value,
            tracked_leg=None,
            tracking_valid=False,
            tracking_reason="not_started",
            visual_knee_angle_deg=None,
            raw_knee_angle_deg=None,
            pose_confidence=0.0,
            inference_fps=0.0,
            frame_index=0,
            error="",
        )

    def _calibration_locked(self) -> bool:
        status = self.calibration.status()
        return status.active or status.state == "ANALYZING"

    @property
    def imu_source(self) -> Any | None:
        with self._imu_lock:
            return self._imu_source

    def list_imu_ports(self) -> list[dict[str, str]]:
        return list_serial_ports()

    def connect_imu(self, port: str, *, baud_rate: int = IMUCAL_BAUD_RATE) -> dict[str, Any]:
        if self._calibration_locked():
            raise RuntimeError("Stop the calibration test before changing the IMU serial connection")
        if self.recorder.status().active:
            raise RuntimeError("Stop recording before changing the IMU serial connection")
        current = self.imu_source
        if current is not None and not bool(getattr(current, "replaceable_by_serial", True)):
            raise RuntimeError(
                "UNO Q Bridge is managed by the Phase 2 launcher and cannot be replaced from the COM controls"
            )
        clean_port = str(port or "").strip()
        if not clean_port:
            raise ValueError("Select a serial port before connecting")
        clean_baud = int(baud_rate)
        if clean_baud <= 0:
            raise ValueError("IMU baud rate must be positive")

        new_source = self._imu_source_factory(clean_port, baud_rate=clean_baud)
        with self._imu_lock:
            previous = self._imu_source
            self._imu_source = new_source
        if previous is not None:
            previous.stop()
        self._reset_imu_estimator("connecting")
        new_source.start()
        wait_until_connected = getattr(new_source, "wait_until_connected", None)
        if callable(wait_until_connected) and not wait_until_connected(timeout_s=5.0):
            new_source.stop()
            with self._imu_lock:
                if self._imu_source is new_source:
                    self._imu_source = None
            raise RuntimeError(f"IMU device did not answer on {clean_port} within 5 seconds")
        start_capture = getattr(new_source, "start_capture", None)
        if callable(start_capture):
            try:
                start_capture(timeout_s=2.0)
            except Exception:
                new_source.stop()
                with self._imu_lock:
                    if self._imu_source is new_source:
                        self._imu_source = None
                raise
        if self._thread is not None and self._thread.is_alive():
            self._ensure_imu_thread()
        return new_source.status()

    def disconnect_imu(self) -> dict[str, Any]:
        if self._calibration_locked():
            raise RuntimeError("Stop the calibration test before disconnecting the IMU serial source")
        if self.recorder.status().active:
            raise RuntimeError("Stop recording before disconnecting the IMU serial source")
        current = self.imu_source
        if current is not None and not bool(getattr(current, "replaceable_by_serial", True)):
            raise RuntimeError(
                "UNO Q Bridge is managed by the Phase 2 launcher; stop the App instead"
            )
        with self._imu_lock:
            source, self._imu_source = self._imu_source, None
        if source is not None:
            source.stop()
        self._reset_imu_estimator("disconnected")
        return _empty_imu_status()

    def reload_imu_filter(self) -> dict[str, Any]:
        if self._calibration_locked():
            raise RuntimeError("Stop the calibration test before reloading the IMU filter configuration")
        if self.recorder.status().active:
            raise RuntimeError("Stop recording before reloading the IMU filter configuration")
        if self._imu_filter_config_path is None:
            raise RuntimeError("No IMU filter configuration path was provided")
        parameters = KneeAngleFilterParameters.load_yaml(self._imu_filter_config_path)
        with self._imu_filter_lock:
            self._imu_filter = KneeAngleFilterMirror(parameters)
            self._latest_imu_estimate = KneeAngleEstimate(reason="filter_reloaded")
            self._acquisition_history.clear()
        return self.imu_estimator_status()

    def imu_estimator_status(self) -> dict[str, Any]:
        with self._imu_filter_lock:
            estimate = self._latest_imu_estimate
            parameters = self._imu_filter.parameters
        payload = estimate.to_record()
        payload.update(
            {
                "valid": estimate.valid,
                "reason": estimate.reason,
                "implementation_id": self._imu_filter.implementation_id,
                "config_path": str(self._imu_filter_config_path or ""),
                "parameters": parameters.to_dict(),
            }
        )
        return payload

    def acquisition_history(self, *, seconds: float = 15.0, max_points: int = 300) -> list[dict[str, Any]]:
        seconds = max(1.0, min(120.0, float(seconds)))
        max_points = max(10, min(1000, int(max_points)))
        cutoff_ns = time.perf_counter_ns() - int(seconds * 1_000_000_000)
        with self._imu_filter_lock:
            rows = [row for row in self._acquisition_history if row["host_monotonic_ns"] >= cutoff_ns]
        if len(rows) <= max_points:
            return rows
        step = max(1, len(rows) // max_points)
        sampled = rows[::step]
        return sampled[-max_points:]

    def _reset_imu_estimator(self, reason: str) -> None:
        with self._imu_filter_lock:
            self._imu_filter.reset()
            self._latest_imu_estimate = KneeAngleEstimate(reason=reason)
            self._acquisition_history.clear()

    def camera_status(self) -> dict[str, Any]:
        profile = self._camera_profile_values()
        return {
            "profile": self._camera_profile,
            "selector": getattr(self.camera, "camera_index", None),
            "active_index": getattr(self.camera, "active_camera_index", None),
            "source": self.camera.description,
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "fps": float(profile["fps"]),
            "reconfigurable": callable(getattr(self.camera, "reconfigure", None))
            and getattr(self.camera, "video_path", None) is None,
            "available_profiles": {
                name: {"width": width, "height": height, "fps": fps}
                for name, (width, height, fps) in CAMERA_PROFILES.items()
            },
        }

    def update_camera_profile(self, profile_name: str) -> dict[str, Any]:
        if self._calibration_locked():
            raise RuntimeError("Stop the calibration test before changing the camera profile")
        if self.recorder.status().active:
            raise RuntimeError("Stop recording before changing the camera profile")
        with self._lock:
            current_state = self._status.state
        if current_state in {"IDLE", "STARTING", "FAILED", "STOPPED"}:
            raise RuntimeError("Wait for the live camera preview before changing the camera profile")
        normalized = str(profile_name or "").strip().lower()
        if normalized not in CAMERA_PROFILES:
            allowed = ", ".join(sorted(CAMERA_PROFILES))
            raise ValueError(f"Unsupported camera profile {profile_name!r}; choose {allowed}")
        reconfigure = getattr(self.camera, "reconfigure", None)
        if not callable(reconfigure) or getattr(self.camera, "video_path", None) is not None:
            raise RuntimeError("Camera profiles are unavailable for this camera source")

        width, height, fps = CAMERA_PROFILES[normalized]
        with self._new_frame:
            previous_state = self._status.state
            self._status = RuntimeStatus(
                **{
                    **asdict(self._status),
                    "state": "CAMERA_RESTARTING",
                    "tracking_valid": False,
                    "tracking_reason": "changing_camera_profile",
                    "error": "",
                }
            )
            self._new_frame.notify_all()
        try:
            reconfigure(width=width, height=height, fps=fps)
        except Exception as exc:
            with self._new_frame:
                self._status = RuntimeStatus(
                    **{
                        **asdict(self._status),
                        "state": previous_state,
                        "tracking_reason": "camera_profile_failed",
                        "error": str(exc),
                    }
                )
                self._new_frame.notify_all()
            raise RuntimeError(f"Could not apply camera profile {normalized}: {exc}") from exc

        self._camera_profile = normalized
        self._reset_tracking_event.set()
        next_state = (
            "AI_FAILED"
            if self._pose_failed.is_set()
            else ("RUNNING" if self._pose_ready.is_set() else "CAMERA_PREVIEW")
        )
        next_reason = "pose_model_failed" if self._pose_failed.is_set() else "camera_profile_changed"
        with self._new_frame:
            self._status = RuntimeStatus(
                **{
                    **asdict(self._status),
                    "state": next_state,
                    "source": self.camera.description,
                    "tracking_reason": next_reason,
                    "error": self._pose_error if self._pose_failed.is_set() else "",
                }
            )
            self._new_frame.notify_all()
        return self.camera_status()

    def _camera_profile_values(self) -> dict[str, float | int]:
        profile_method = getattr(self.camera, "profile", None)
        if callable(profile_method):
            return profile_method()
        return {
            "width": int(getattr(self.camera, "width", 0) or 0),
            "height": int(getattr(self.camera, "height", 0) or 0),
            "fps": float(getattr(self.camera, "fps", 0.0) or 0.0),
        }

    def _infer_camera_profile(self) -> str:
        profile = self._camera_profile_values()
        current = (int(profile["width"]), int(profile["height"]), float(profile["fps"]))
        for name, values in CAMERA_PROFILES.items():
            if current == values:
                return name
        return "custom"


    def _ensure_imu_thread(self) -> None:
        with self._imu_lock:
            if self._imu_thread is not None and self._imu_thread.is_alive():
                return
            self._imu_thread = threading.Thread(
                target=self._imu_loop,
                name="vision-imu-recorder-pump",
                daemon=True,
            )
            self._imu_thread.start()

    def _activate_existing_imu_source(self, source: Any) -> None:
        try:
            wait_until_connected = getattr(source, "wait_until_connected", None)
            if callable(wait_until_connected) and not wait_until_connected(timeout_s=5.0):
                return
            start_capture = getattr(source, "start_capture", None)
            if callable(start_capture) and not bool(source.status().get("capture_active")):
                start_capture(timeout_s=2.0)
        except Exception:
            return

    def start(self) -> None:
        """Start camera preview and pose initialization asynchronously."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._pose_ready.clear()
            self._pose_failed.clear()
            self._pose_error = ""
            source = self.imu_source
            if source is not None:
                source.start()
                threading.Thread(
                    target=self._activate_existing_imu_source,
                    args=(source,),
                    name="vision-imu-source-activate",
                    daemon=True,
                ).start()
            self._ensure_imu_thread()
            self._status = RuntimeStatus(
                **{
                    **asdict(self._status),
                    "state": "STARTING",
                    "tracking_reason": "initializing_camera",
                    "error": "",
                }
            )
            self._thread = threading.Thread(
                target=self._initialize_and_run,
                name="vision-imu-runtime",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        timer_cancel = self._calibration_timer_cancel
        if timer_cancel is not None:
            timer_cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))
        self.recorder.close()
        source = self.imu_source
        if source is not None:
            source.stop()
        imu_thread = self._imu_thread
        if imu_thread is not None and imu_thread.is_alive():
            imu_thread.join(timeout=1.0)
        self.estimator.close()
        self.camera.close()
        with self._new_frame:
            if self._status.state != "FAILED":
                self._status = RuntimeStatus(**{**asdict(self._status), "state": "STOPPED"})
            self._new_frame.notify_all()

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._status)
            payload["tuning"] = self._tuning.to_dict()
            payload["pose_backend"] = getattr(self.estimator, "description", type(self.estimator).__name__)
        recording_payload = asdict(self.recorder.status())
        recording_payload.update(self._recording_export_payload(recording_payload))
        payload["recording"] = recording_payload
        source = self.imu_source
        payload["imu"] = source.status() if source is not None else _empty_imu_status()
        payload["imu_estimator"] = self.imu_estimator_status()
        payload["camera"] = self.camera_status()
        calibration_payload = asdict(self.calibration.status())
        calibration_payload.update(self._calibration_export_payload(calibration_payload))
        payload["calibration"] = calibration_payload
        return payload

    def _recording_export_payload(self, status: Mapping[str, Any]) -> dict[str, str]:
        archive = _optional_owned_path(
            status.get("archive_path"), self.recording_root, require_exists=False
        )
        if archive is None:
            return {
                "display_path": self.recording_root.name,
                "download_url": "",
                "browse_url": "/exports/recordings/",
            }
        relative = archive.relative_to(self.recording_root)
        return {
            "display_path": f"{self.recording_root.name}/{relative.as_posix()}",
            "download_url": f"/exports/recordings/{quote(relative.as_posix())}",
            "browse_url": "/exports/recordings/",
        }

    def _calibration_export_payload(self, status: Mapping[str, Any]) -> dict[str, str]:
        run_directory = _optional_owned_path(
            status.get("run_directory"), self.calibration_root, directory=True
        )
        recommendation = _optional_owned_path(
            status.get("recommendation_path"), self.calibration_root
        )
        if run_directory is None:
            browse_url = "/exports/calibration/"
            display_path = self.calibration_root.name
        else:
            run_relative = run_directory.relative_to(self.calibration_root)
            browse_url = f"/exports/calibration/{quote(run_relative.as_posix())}/"
            display_path = f"{self.calibration_root.name}/{run_relative.as_posix()}"
        recommendation_display = ""
        recommendation_url = ""
        if recommendation is not None:
            relative = recommendation.relative_to(self.calibration_root)
            recommendation_display = (
                f"{self.calibration_root.name}/{relative.as_posix()}"
            )
            recommendation_url = f"/exports/calibration/{quote(relative.as_posix())}"
        return {
            "display_path": display_path,
            "browse_url": browse_url,
            "recommendation_display_path": recommendation_display,
            "recommendation_url": recommendation_url,
        }

    def tuning(self) -> dict[str, Any]:
        with self._lock:
            return self._tuning.to_dict()

    def update_tuning(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._calibration_locked():
            raise RuntimeError("Stop the calibration test before changing pose-reference tuning")
        with self._lock:
            self._tuning = RuntimeTuning.from_mapping(payload, current=self._tuning)
            self._status = RuntimeStatus(
                **{
                    **asdict(self._status),
                    "requested_leg": self._tuning.leg_side.value,
                }
            )
            return self._tuning.to_dict()

    @staticmethod
    def _format_imu_description(status: Mapping[str, Any]) -> str:
        port = str(status.get("port", "") or "")
        baud_rate = int(status.get("baud_rate", 0) or 0)
        return f"{port}@{baud_rate}" if port and baud_rate > 0 else port

    def start_recording(self) -> dict[str, Any]:
        if self._calibration_locked():
            raise RuntimeError("A calibration test is already recording")
        with self._lock:
            tuning = self._tuning.to_dict()
            tuning["pose_backend"] = getattr(self.estimator, "description", type(self.estimator).__name__)
            source = self.camera.description
            imu_description = ""
            imu_source = self.imu_source
            if imu_source is not None:
                imu_status = imu_source.status()
                imu_description = self._format_imu_description(imu_status)
        tuning["imu_filter"] = self.imu_estimator_status()
        self.recorder.request_start(
            settings=tuning,
            source=source,
            imu_source=imu_description,
        )
        if imu_source is not None and imu_source.configured:
            imu_status = imu_source.status()
            if not imu_status.get("capture_active"):
                try:
                    imu_source.start_capture()
                except Exception:
                    self.recorder.stop()
                    raise
        return self.status()["recording"]

    def start_calibration_test(self, test_type: str) -> dict[str, Any]:
        with self._calibration_action_lock:
            if self.recorder.status().active:
                raise RuntimeError("Stop the current recording before starting calibration")
            status = self.status()
            imu = status.get("imu", {})
            if not imu.get("connected") or not imu.get("capture_active"):
                raise RuntimeError("Connect the IMU source before starting calibration")
            if not imu.get("thigh_ready") or not imu.get("shank_ready"):
                raise RuntimeError("Both thigh and shank IMUs must be READY")
            if not status.get("tracking_valid"):
                raise RuntimeError("Vision tracking must be VALID before the test starts")

            with self._imu_filter_lock:
                base_parameters = self._imu_filter.parameters
            prepared = self.calibration.prepare_test(
                test_type,
                base_parameters=base_parameters,
            )
            with self._lock:
                tuning = self._tuning.to_dict()
                tuning["pose_backend"] = getattr(
                    self.estimator, "description", type(self.estimator).__name__
                )
                source_description = self.camera.description
            imu_description = self._format_imu_description(imu)
            tuning["imu_filter"] = self.imu_estimator_status()
            run_directory = Path(prepared["run_directory"])
            try:
                self.recorder.request_start(
                    settings=tuning,
                    source=source_description,
                    imu_source=imu_description,
                    output_root=run_directory,
                    session_name=Path(str(prepared["archive_name"])).stem,
                    extra_metadata={
                        "recording_purpose": "vision_imu_filter_calibration",
                        "calibration_test_type": prepared["test_type"],
                        "automatic_duration_s": prepared["duration_s"],
                    },
                )
            except Exception as exc:
                try:
                    self.recorder.stop()
                except Exception:
                    pass
                self.calibration.fail(f"Could not start calibration recording: {exc}")
                raise

            timer_cancel = threading.Event()
            previous_cancel = self._calibration_timer_cancel
            if previous_cancel is not None:
                previous_cancel.set()
            self._calibration_timer_cancel = timer_cancel
            timer_thread = threading.Thread(
                target=self._calibration_auto_stop,
                args=(prepared["test_type"], float(prepared["duration_s"]), timer_cancel),
                name=f"vision-imu-{prepared['test_type']}-calibration-timer",
                daemon=True,
            )
            self._calibration_timer_thread = timer_thread
            timer_thread.start()
            return self.status()["calibration"]

    def stop_calibration_test(self) -> dict[str, Any]:
        with self._calibration_action_lock:
            calibration_status = self.calibration.status()
            if not calibration_status.active:
                raise RuntimeError("No calibration test is currently running")
            timer_cancel = self._calibration_timer_cancel
            if timer_cancel is not None:
                timer_cancel.set()
            recording_status = self.recorder.stop()
            if recording_status.error:
                self.calibration.fail(recording_status.error)
                return self.status()["calibration"]
            self.calibration.finish_test(
                calibration_status.current_test,
                recording_status.archive_path,
            )
            return self.status()["calibration"]

    def cancel_calibration_test(self) -> dict[str, Any]:
        with self._calibration_action_lock:
            timer_cancel = self._calibration_timer_cancel
            if timer_cancel is not None:
                timer_cancel.set()
            if self.recorder.status().active:
                self.recorder.stop()
            self.calibration.cancel_test()
            return self.status()["calibration"]

    def open_calibration_location(self) -> dict[str, str]:
        status = self._calibration_export_payload(asdict(self.calibration.status()))
        return {"path": status["display_path"], "url": status["browse_url"]}

    def _calibration_auto_stop(
        self,
        test_type: str,
        duration_s: float,
        cancel_event: threading.Event,
    ) -> None:
        if cancel_event.wait(max(0.0, duration_s)):
            return
        try:
            current = self.calibration.status()
            if current.active and current.current_test == test_type:
                self.stop_calibration_test()
        except Exception as exc:
            self.calibration.fail(str(exc), stage=test_type)

    def stop_recording(self) -> dict[str, Any]:
        if self.calibration.status().active:
            raise RuntimeError("Use the calibration stop control for an active calibration test")
        self.recorder.stop()
        return self.status()["recording"]

    def open_recordings_location(self) -> dict[str, str]:
        return {"path": self.recording_root.name, "url": "/exports/recordings/"}

    def wait_for_jpeg(self, after_sequence: int, timeout_s: float = 1.0) -> tuple[int, bytes | None]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._new_frame:
            while self._frame_sequence <= after_sequence and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._new_frame.wait(timeout=remaining)
            return self._frame_sequence, self._latest_jpeg

    def _initialize_and_run(self) -> None:
        capture_thread: threading.Thread | None = None
        try:
            self.camera.open()
            with self._new_frame:
                self._status = RuntimeStatus(
                    **{
                        **asdict(self._status),
                        "state": "CAMERA_PREVIEW",
                        "source": self.camera.description,
                        "tracking_reason": "initializing_pose_model",
                        "error": "",
                    }
                )
                self._new_frame.notify_all()

            capture_thread = threading.Thread(
                target=self._capture_loop,
                name="vision-imu-camera",
                daemon=True,
            )
            self._capture_thread = capture_thread
            capture_thread.start()

            try:
                self.estimator.open()
            except Exception as exc:
                self._pose_error = str(exc)
                self._pose_failed.set()
                with self._new_frame:
                    self._status = RuntimeStatus(
                        **{
                            **asdict(self._status),
                            "state": "AI_FAILED",
                            "tracking_valid": False,
                            "tracking_reason": "pose_model_failed",
                            "error": self._pose_error,
                        }
                    )
                    self._new_frame.notify_all()
            else:
                self._pose_ready.set()

            self._stop_event.wait()
        except Exception as exc:
            with self._new_frame:
                self._status = RuntimeStatus(
                    **{
                        **asdict(self._status),
                        "state": "FAILED",
                        "tracking_valid": False,
                        "tracking_reason": "camera_runtime_failure",
                        "error": str(exc),
                    }
                )
                self._new_frame.notify_all()
            self._stop_event.set()
        finally:
            self._stop_event.set()
            if capture_thread is not None and capture_thread.is_alive():
                capture_thread.join(timeout=2.0)
            self.recorder.close()
            source = self.imu_source
            if source is not None:
                source.stop()
            self.estimator.close()
            self.camera.close()

    def _imu_loop(self) -> None:
        while not self._stop_event.is_set():
            source = self.imu_source
            if source is None:
                time.sleep(0.05)
                continue
            sample = source.get_sample(timeout_s=0.1)
            if sample is None:
                continue
            with self._imu_filter_lock:
                estimate = self._imu_filter.update(sample)
                self._latest_imu_estimate = estimate
            self.recorder.write_imu(sample, estimate=estimate)

    def _capture_loop(self) -> None:
        cv2 = self._load_cv2()
        frame_index = 0
        ema_fps = 0.0
        last_completed = time.perf_counter()
        smoothed_angle: float | None = None
        smoothed_leg: LegSide | None = None
        landmark_filter = LegLandmarkEmaFilter()
        active_rotation: FrameRotation | None = None

        try:
            while not self._stop_event.is_set():
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                host_monotonic_ns = time.perf_counter_ns()
                timestamp_ms = host_monotonic_ns // 1_000_000
                with self._lock:
                    tuning = self._tuning

                if self._reset_tracking_event.is_set():
                    self._reset_tracking_event.clear()
                    active_rotation = None
                    landmark_filter.reset()
                    smoothed_angle = None
                    smoothed_leg = None
                    reset_tracking = getattr(self.estimator, "reset_tracking", None)
                    if callable(reset_tracking):
                        reset_tracking()

                if active_rotation is not tuning.frame_rotation:
                    active_rotation = tuning.frame_rotation
                    landmark_filter.reset()
                    smoothed_angle = None
                    smoothed_leg = None
                    reset_tracking = getattr(self.estimator, "reset_tracking", None)
                    if callable(reset_tracking):
                        reset_tracking()
                frame = self._rotate_frame(frame, tuning.frame_rotation, cv2)

                leg = None
                raw_angle = KneeAngleResult(None, False, 0.0, "initializing_pose_model")
                filtered_angle = raw_angle
                tracked_leg = None

                if self._pose_ready.is_set() and not self._pose_failed.is_set():
                    try:
                        pose_frame = self.estimator.estimate(frame, timestamp_ms)
                        raw_leg = pose_frame.leg(tuning.leg_side) if pose_frame is not None else None
                        image_aspect_ratio = float(frame.shape[1]) / float(frame.shape[0])
                        raw_angle = calculate_knee_flexion_deg(
                            raw_leg,
                            confidence_threshold=tuning.confidence_threshold,
                            min_segment_length=tuning.min_segment_length,
                            image_aspect_ratio=image_aspect_ratio,
                        )

                        if raw_leg is None or not raw_angle.valid:
                            leg = None
                            landmark_filter.reset()
                            filtered_angle = raw_angle
                            smoothed_angle = None
                            smoothed_leg = None
                        else:
                            if smoothed_leg is not raw_leg.side:
                                landmark_filter.reset()
                                smoothed_angle = None
                                smoothed_leg = raw_leg.side
                            leg = landmark_filter.apply(
                                raw_leg,
                                smoothing=tuning.landmark_smoothing,
                            )
                            smoothed_geometry = calculate_knee_flexion_deg(
                                leg,
                                confidence_threshold=tuning.confidence_threshold,
                                min_segment_length=tuning.min_segment_length,
                                image_aspect_ratio=image_aspect_ratio,
                            )
                            filtered_angle, smoothed_angle = self._filter_angle(
                                smoothed_geometry,
                                previous=smoothed_angle,
                                smoothing=tuning.angle_smoothing,
                                max_jump_deg=tuning.max_angle_jump_deg,
                            )
                        tracked_leg = raw_leg.side.value if raw_leg is not None else None
                    except Exception as exc:
                        self._pose_error = str(exc)
                        self._pose_failed.set()
                        self._pose_ready.clear()
                        leg = None
                        landmark_filter.reset()
                        raw_angle = KneeAngleResult(None, False, 0.0, "pose_inference_failed")
                        filtered_angle = raw_angle
                        smoothed_angle = None
                        smoothed_leg = None
                elif self._pose_failed.is_set():
                    raw_angle = KneeAngleResult(None, False, 0.0, "pose_model_failed")
                    filtered_angle = raw_angle

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(1e-6, now - last_completed)
                last_completed = now
                if self._pose_ready.is_set() and not self._pose_failed.is_set():
                    ema_fps = (
                        instantaneous_fps
                        if ema_fps <= 0
                        else (0.9 * ema_fps + 0.1 * instantaneous_fps)
                    )
                else:
                    ema_fps = 0.0

                annotated = draw_leg_overlay(
                    frame,
                    leg,
                    filtered_angle,
                    inference_fps=ema_fps,
                    source_description=self.camera.description,
                    recording_active=self.recorder.status().active,
                )
                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
                if not ok:
                    continue

                frame_index += 1
                with self._imu_filter_lock:
                    imu_estimate = self._latest_imu_estimate
                difference_deg = None
                if (
                    filtered_angle.valid
                    and filtered_angle.angle_deg is not None
                    and imu_estimate.valid
                    and imu_estimate.knee_angle_deg is not None
                ):
                    difference_deg = float(filtered_angle.angle_deg) - float(imu_estimate.knee_angle_deg)
                sample = {
                    "host_monotonic_ns": host_monotonic_ns,
                    "frame_index": frame_index,
                    "visual_knee_angle_deg": filtered_angle.angle_deg,
                    "raw_knee_angle_deg": raw_angle.angle_deg,
                    "pose_confidence": filtered_angle.confidence,
                    "tracking_valid": filtered_angle.valid,
                    "tracking_reason": filtered_angle.reason,
                    "tracked_leg": tracked_leg,
                    "inference_fps": ema_fps,
                    "hip_x": leg.hip.x if leg is not None else None,
                    "hip_y": leg.hip.y if leg is not None else None,
                    "knee_x": leg.knee.x if leg is not None else None,
                    "knee_y": leg.knee.y if leg is not None else None,
                    "ankle_x": leg.ankle.x if leg is not None else None,
                    "ankle_y": leg.ankle.y if leg is not None else None,
                    "live_imu_filter_valid": imu_estimate.valid,
                    "live_imu_knee_angle_deg": imu_estimate.knee_angle_deg,
                    "live_imu_knee_speed_deg_s": imu_estimate.knee_speed_deg_s,
                    "live_angle_difference_deg": difference_deg,
                }
                with self._imu_filter_lock:
                    self._acquisition_history.append(
                        {
                            "host_monotonic_ns": host_monotonic_ns,
                            "visual_knee_angle_deg": filtered_angle.angle_deg,
                            "visual_valid": filtered_angle.valid,
                            "imu_knee_angle_deg": imu_estimate.knee_angle_deg,
                            "imu_valid": imu_estimate.valid,
                            "angle_difference_deg": difference_deg,
                        }
                    )
                self.recorder.write(annotated, preferred_fps=ema_fps or 15.0, sample=sample)

                if self._pose_failed.is_set():
                    state = "AI_FAILED"
                    error = self._pose_error
                elif self._pose_ready.is_set():
                    state = "RUNNING"
                    error = ""
                else:
                    state = "CAMERA_PREVIEW"
                    error = ""

                with self._new_frame:
                    self._latest_jpeg = encoded.tobytes()
                    self._frame_sequence += 1
                    self._status = RuntimeStatus(
                        state=state,
                        source=self.camera.description,
                        requested_leg=tuning.leg_side.value,
                        tracked_leg=tracked_leg,
                        tracking_valid=filtered_angle.valid,
                        tracking_reason=filtered_angle.reason,
                        visual_knee_angle_deg=filtered_angle.angle_deg,
                        raw_knee_angle_deg=raw_angle.angle_deg,
                        pose_confidence=filtered_angle.confidence,
                        inference_fps=ema_fps,
                        frame_index=frame_index,
                        error=error,
                    )
                    self._new_frame.notify_all()
        except Exception as exc:
            with self._new_frame:
                self._status = RuntimeStatus(
                    **{
                        **asdict(self._status),
                        "state": "FAILED",
                        "tracking_valid": False,
                        "tracking_reason": "camera_runtime_failure",
                        "error": str(exc),
                    }
                )
                self._new_frame.notify_all()
            self._stop_event.set()


    @staticmethod
    def _rotate_frame(frame: Any, rotation: FrameRotation, cv2: Any) -> Any:
        if rotation is FrameRotation.NONE:
            return frame
        if rotation is FrameRotation.CLOCKWISE_90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation is FrameRotation.COUNTERCLOCKWISE_90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotation is FrameRotation.ROTATE_180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        raise ValueError(f"Unsupported frame rotation: {rotation}")

    @staticmethod
    def _filter_angle(
        angle: KneeAngleResult,
        *,
        previous: float | None,
        smoothing: float,
        max_jump_deg: float,
    ) -> tuple[KneeAngleResult, float | None]:
        if not angle.valid or angle.angle_deg is None:
            return angle, None
        raw = float(angle.angle_deg)
        if previous is not None and abs(raw - previous) > max_jump_deg:
            return KneeAngleResult(None, False, angle.confidence, "angle_jump"), previous
        filtered = raw if previous is None else smoothing * previous + (1.0 - smoothing) * raw
        return KneeAngleResult(filtered, True, angle.confidence, "ok"), filtered

    @staticmethod
    def _load_cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed") from exc
        return cv2
