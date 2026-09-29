from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .calibration_workflow import CalibrationWorkflow
from .final_protocol import FINAL_PROTOCOL_VERSION, PROTOCOL_SUMMARY
from .camera_calibration import CheckerboardCalibrationSession
from .camera_source import OpenCvCameraSource
from .imu_filter import (
    KneeAngleEstimate,
    KneeAngleFilterMirror,
    KneeAngleFilterParameters,
)
from .imu_serial import IMUCAL_BAUD_RATE, SerialImuSource, list_serial_ports
from .marker_reference import AprilTagReference, MarkerReferenceResult
from .overlay import draw_marker_overlay
from .provenance import load_runtime_provenance
from .recorder import AnnotatedVideoRecorder
from .reference_setup import ReferenceSetupStore
from .reference_validation import KnownAngleReferenceValidationSession
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
    tracking_valid: bool
    tracking_reason: str
    visual_knee_angle_deg: float | None
    marker_quality: float
    thigh_tag_detected: bool
    shank_tag_detected: bool
    processing_fps: float
    frame_index: int
    error: str
    thigh_pose_valid: bool = False
    shank_pose_valid: bool = False
    thigh_reprojection_rmse_px: float | None = None
    shank_reprojection_rmse_px: float | None = None
    thigh_view_angle_deg: float | None = None
    shank_view_angle_deg: float | None = None


class VisionRuntime:
    """Own camera, AprilTag reference, annotated stream, IMU path, and recorder."""

    def __init__(
        self,
        camera: OpenCvCameraSource,
        reference: AprilTagReference,
        *,
        recording_root: str | Path = "video_recordings",
        calibration_root: str | Path = "calibration_suggestions",
        imu_source: Any | None = None,
        imu_source_factory: Any = SerialImuSource,
        imu_filter: KneeAngleFilterMirror | None = None,
        imu_filter_config_path: str | Path | None = None,
        reference_setup_root: str | Path | None = None,
    ) -> None:
        self.camera = camera
        self.reference = reference
        self._tuning = RuntimeTuning()
        self.recording_root = Path(recording_root).expanduser().resolve()
        self.calibration_root = Path(calibration_root).expanduser().resolve()
        self.reference_setup_root = (
            Path(reference_setup_root).expanduser().resolve()
            if reference_setup_root is not None
            else self.calibration_root.parent / "reference_setup"
        )
        self.recording_root.mkdir(parents=True, exist_ok=True)
        self.calibration_root.mkdir(parents=True, exist_ok=True)
        self.reference_setup_root.mkdir(parents=True, exist_ok=True)
        self.recorder = AnnotatedVideoRecorder(self.recording_root)
        self.calibration = CalibrationWorkflow(self.calibration_root)
        self.camera_calibration = CheckerboardCalibrationSession()
        self.reference_validation = KnownAngleReferenceValidationSession()
        self.reference_setup_store = ReferenceSetupStore(self.reference_setup_root)
        self._provenance = load_runtime_provenance()
        self._reference_setup_status: dict[str, Any] = {
            "persistent": True,
            "persistent_state": "NOT_LOADED",
            "persistent_message": "Persistent camera setup has not been checked yet.",
            "path": str(self.reference_setup_store.yaml_path),
            "log_path": str(self.reference_setup_store.log_path),
            "saved_utc": "",
            "persistent_rms_reprojection_error_px": None,
            "marker_reference_validation": {},
        }
        # Final scientific calibration workflow is active.
        self._calibration_workflow_ready = True
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
        self._reference_ready = threading.Event()
        self._reference_failed = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._imu_thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_sequence = 0
        self._reference_error = ""
        self._reset_tracking_event = threading.Event()
        self._camera_profile = self._infer_camera_profile()
        self._status = RuntimeStatus(
            state="IDLE",
            source=self.camera.description,
            tracking_valid=False,
            tracking_reason="not_started",
            visual_knee_angle_deg=None,
            marker_quality=0.0,
            thigh_tag_detected=False,
            shank_tag_detected=False,
            processing_fps=0.0,
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
        if self.reference_validation.active:
            raise RuntimeError("Cancel known-angle reference validation before changing the camera profile")
        if self.camera_calibration.status()["active"]:
            raise RuntimeError("Cancel checkerboard camera calibration before changing the camera profile")
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
        clear_camera_calibration = getattr(self.reference, "clear_camera_calibration", None)
        if callable(clear_camera_calibration):
            clear_camera_calibration()
        self.camera_calibration.invalidate_for_camera_change()
        self._load_persistent_reference_setup()
        self._reset_tracking_event.set()
        next_state = (
            "REFERENCE_FAILED"
            if self._reference_failed.is_set()
            else ("RUNNING" if self._reference_ready.is_set() else "CAMERA_PREVIEW")
        )
        next_reason = (
            "reference_backend_failed" if self._reference_failed.is_set() else "camera_profile_changed"
        )
        with self._new_frame:
            self._status = RuntimeStatus(
                **{
                    **asdict(self._status),
                    "state": next_state,
                    "source": self.camera.description,
                    "tracking_reason": next_reason,
                    "error": self._reference_error if self._reference_failed.is_set() else "",
                }
            )
            self._new_frame.notify_all()
        return self.camera_status()

    def start_camera_calibration(self) -> dict[str, Any]:
        if self.reference_validation.active:
            raise RuntimeError("Cancel known-angle reference validation before camera calibration")
        if self._calibration_locked():
            raise RuntimeError("Stop the IMU calibration test before camera calibration")
        if self.recorder.status().active:
            raise RuntimeError("Stop recording before camera calibration")
        with self._lock:
            if self._status.state in {"IDLE", "STARTING", "FAILED", "STOPPED"}:
                raise RuntimeError("Wait for the live camera preview before camera calibration")
        return self.camera_calibration.start()

    def cancel_camera_calibration(self) -> dict[str, Any]:
        return self.camera_calibration.cancel()

    def reference_validation_status(self) -> dict[str, Any]:
        live = self.reference_validation.status()
        stored = self.reference_setup_store.marker_reference_validation()
        if live.get("state") == "IDLE" and stored:
            metrics = stored.get("metrics") if isinstance(stored, Mapping) else None
            passed = bool(isinstance(metrics, Mapping) and metrics.get("passed"))
            live = {
                **live,
                "state": "PERSISTED_PASSED" if passed else "PERSISTED_FAILED",
                "message": (
                    "Known-angle reference validation is saved and passed."
                    if passed
                    else "The saved known-angle validation did not pass; repeat it before dynamic calibration."
                ),
                "result": stored,
                "progress": 1.0,
            }
        live["persisted"] = stored
        live["persisted_passed"] = self._reference_validation_passed(stored)
        return live

    @staticmethod
    def _reference_validation_passed(result: Mapping[str, Any] | None) -> bool:
        if not isinstance(result, Mapping):
            return False
        metrics = result.get("metrics")
        return bool(isinstance(metrics, Mapping) and metrics.get("passed") is True)

    def start_reference_validation(self) -> dict[str, Any]:
        if self.camera_calibration.status()["active"]:
            raise RuntimeError("Finish or cancel checkerboard camera calibration first")
        if self._calibration_locked() or self.recorder.status().active:
            raise RuntimeError("Stop recording/calibration before reference validation")
        if self._reference_setup_status.get("persistent_state") != "LOADED":
            raise RuntimeError("Complete and save checkerboard camera calibration first")
        return self.reference_validation.start()

    def capture_reference_validation(self) -> dict[str, Any]:
        if not self.reference_validation.active:
            raise RuntimeError("Start known-angle reference validation first")
        status = self.status()
        if not status.get("tracking_valid"):
            raise RuntimeError("Both tags must be visible and Reference must be VALID")
        rows = self.acquisition_history(seconds=2.0, max_points=120)
        result = self.reference_validation.capture(rows)
        completed = result.get("result")
        if isinstance(completed, Mapping):
            self.reference_setup_store.save_marker_reference_validation(completed)
            self._load_persistent_reference_setup()
        return self.reference_validation_status()

    def cancel_reference_validation(self) -> dict[str, Any]:
        return self.reference_validation.cancel()

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


    def _reference_frame_dimensions(self) -> tuple[int, int]:
        profile = self._camera_profile_values()
        width = int(profile["width"])
        height = int(profile["height"])
        if self._tuning.frame_rotation in {FrameRotation.CLOCKWISE_90, FrameRotation.COUNTERCLOCKWISE_90}:
            width, height = height, width
        return width, height

    def _load_persistent_reference_setup(self) -> None:
        width, height = self._reference_frame_dimensions()
        load = self.reference_setup_store.load_camera_calibration(
            expected_width=width,
            expected_height=height,
            expected_frame_rotation=self._tuning.frame_rotation.value,
        )
        configure_camera = getattr(self.reference, "configure_camera", None)
        clear_camera = getattr(self.reference, "clear_camera_calibration", None)
        if load.valid and callable(configure_camera):
            configure_camera(load.calibration)
        elif callable(clear_camera):
            clear_camera()
        stored_validation = self.reference_setup_store.marker_reference_validation()
        self._reference_setup_status = {
            "persistent": True,
            "persistent_state": load.state,
            "persistent_message": load.message,
            "path": str(self.reference_setup_store.yaml_path),
            "log_path": str(self.reference_setup_store.log_path),
            "saved_utc": load.saved_utc,
            "persistent_rms_reprojection_error_px": load.rms_reprojection_error_px,
            "marker_reference_validation": stored_validation,
            "reference_setup_sha256": self.reference_setup_store.current_sha256(),
        }

    def _persist_reference_setup(self) -> None:
        result = self.camera_calibration.result()
        if result is None:
            return
        try:
            self.reference_setup_store.save_camera_calibration(
                result,
                camera_description=self.camera.description,
                camera_fps=float(getattr(self.camera, "fps", 0.0) or 0.0),
                tag_size_m=float(getattr(self.reference, "tag_size_m", 0.070)),
                thigh_tag_id=int(getattr(self.reference, "thigh_tag_id", 0)),
                shank_tag_id=int(getattr(self.reference, "shank_tag_id", 1)),
                specification=self.camera_calibration.specification,
                frame_rotation=self._tuning.frame_rotation.value,
            )
        except Exception as exc:
            self._reference_setup_status = {
                "persistent": True,
                "persistent_state": "SAVE_FAILED",
                "persistent_message": f"Camera calibration is active but could not be persisted: {exc}",
                "path": str(self.reference_setup_store.yaml_path),
                "log_path": str(self.reference_setup_store.log_path),
                "saved_utc": "",
                "persistent_rms_reprojection_error_px": result.rms_reprojection_error_px,
                "marker_reference_validation": {},
            }
            return
        self._reference_setup_status = {
            "persistent": True,
            "persistent_state": "SAVED",
            "persistent_message": "Camera calibration saved persistently and applied to the 3-D reference.",
            "path": str(self.reference_setup_store.yaml_path),
            "log_path": str(self.reference_setup_store.log_path),
            "saved_utc": "",
            "persistent_rms_reprojection_error_px": result.rms_reprojection_error_px,
        }
        # Re-read the just-written file so status contains the canonical saved timestamp
        # and the same validation path used on subsequent app starts.
        self._load_persistent_reference_setup()

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
        """Start camera, tag reference, IMU stream, and web-facing runtime."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._reference_ready.clear()
            self._reference_failed.clear()
            self._reference_error = ""
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
        self.reference.close()
        self.camera.close()
        with self._new_frame:
            if self._status.state != "FAILED":
                self._status = RuntimeStatus(**{**asdict(self._status), "state": "STOPPED"})
            self._new_frame.notify_all()

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._status)
            payload["tuning"] = self._tuning.to_dict()
            payload["reference_backend"] = getattr(
                self.reference, "description", type(self.reference).__name__
            )
            reference_configuration = getattr(self.reference, "configuration_status", None)
            reference_details = (
                reference_configuration() if callable(reference_configuration) else {}
            )
            payload["reference"] = {
                "type": "apriltag",
                "dictionary": "DICT_APRILTAG_36h11",
                "thigh_tag_id": int(getattr(self.reference, "thigh_tag_id", 0)),
                "shank_tag_id": int(getattr(self.reference, "shank_tag_id", 1)),
                "setup_required": not bool(reference_details.get("camera_calibrated", False)),
                **reference_details,
            }
        recording_payload = asdict(self.recorder.status())
        recording_payload.update(self._recording_export_payload(recording_payload))
        payload["recording"] = recording_payload
        source = self.imu_source
        payload["imu"] = source.status() if source is not None else _empty_imu_status()
        payload["imu_estimator"] = self.imu_estimator_status()
        payload["camera"] = self.camera_status()
        camera_calibration_payload = self.camera_calibration.status()
        camera_calibration_payload.update(self._reference_setup_status)
        camera_calibration_payload["persistent"] = True
        payload["camera_calibration"] = camera_calibration_payload
        payload["reference_validation"] = self.reference_validation_status()
        payload["protocol"] = {
            "version": FINAL_PROTOCOL_VERSION,
            "summary": PROTOCOL_SUMMARY,
        }
        payload["provenance"] = self._provenance
        calibration_payload = asdict(self.calibration.status())
        calibration_payload.update(self._calibration_export_payload(calibration_payload))
        # Expose whether the scientific calibration workflow is enabled.
        calibration_payload["workflow_ready"] = self._calibration_workflow_ready
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
        if self.reference_validation.active:
            raise RuntimeError("Cancel known-angle reference validation before changing camera presentation settings")
        if self.camera_calibration.status()["active"]:
            raise RuntimeError("Cancel checkerboard camera calibration before changing camera presentation settings")
        if self._calibration_locked():
            raise RuntimeError("Stop the calibration test before changing camera presentation settings")
        with self._lock:
            previous_rotation = self._tuning.frame_rotation
            self._tuning = RuntimeTuning.from_mapping(payload, current=self._tuning)
            rotation_changed = self._tuning.frame_rotation is not previous_rotation
            if rotation_changed:
                clear_camera_calibration = getattr(self.reference, "clear_camera_calibration", None)
                if callable(clear_camera_calibration):
                    clear_camera_calibration()
                self.camera_calibration.invalidate_for_camera_change()
                self._load_persistent_reference_setup()
            self._reset_tracking_event.set()
            return self._tuning.to_dict()

    @staticmethod
    def _format_imu_description(status: Mapping[str, Any]) -> str:
        port = str(status.get("port", "") or "")
        baud_rate = int(status.get("baud_rate", 0) or 0)
        return f"{port}@{baud_rate}" if port and baud_rate > 0 else port

    def start_recording(self) -> dict[str, Any]:
        if self.reference_validation.active:
            raise RuntimeError("Finish or cancel known-angle reference validation before recording")
        if self.camera_calibration.status()["active"]:
            raise RuntimeError("Finish or cancel checkerboard camera calibration before recording")
        if self._calibration_locked():
            raise RuntimeError("A calibration test is already recording")
        with self._lock:
            tuning = self._tuning.to_dict()
            tuning["reference_backend"] = getattr(
                self.reference, "description", type(self.reference).__name__
            )
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

    def _write_calibration_run_provenance(
        self,
        run_directory: Path,
        *,
        base_parameters: KneeAngleFilterParameters,
    ) -> None:
        snapshot = self.reference_setup_store.snapshot_to(
            run_directory / "reference_setup_snapshot.yaml"
        )
        payload = {
            "schema_version": 1,
            "protocol_version": FINAL_PROTOCOL_VERSION,
            "protocol": PROTOCOL_SUMMARY,
            "runtime_provenance": self._provenance,
            "reference_setup_snapshot": snapshot,
            "active_reference_setup_sha256": self.reference_setup_store.current_sha256(),
            "known_angle_reference_validation": self.reference_setup_store.marker_reference_validation(),
            "camera_source": self.camera.description,
            "imu_filter_implementation_id": self._imu_filter.implementation_id,
            "baseline_filter_parameters": base_parameters.to_dict(),
        }
        (run_directory / "RUN_PROVENANCE.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _assert_calibration_run_provenance_current(self, run_directory: Path) -> None:
        path = run_directory / "RUN_PROVENANCE.json"
        if not path.is_file():
            raise RuntimeError("Calibration run provenance is missing; repeat the static stage")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("Calibration run provenance is unreadable; repeat the static stage") from exc
        expected = str(payload.get("active_reference_setup_sha256", "") or "")
        current = self.reference_setup_store.current_sha256()
        if not expected or current != expected:
            raise RuntimeError(
                "Reference setup changed after the static stage; repeat the static stage before dynamic calibration"
            )

    def start_calibration_test(self, test_type: str) -> dict[str, Any]:
        with self._calibration_action_lock:
            if self.reference_validation.active:
                raise RuntimeError("Finish or cancel known-angle reference validation first")
            if self.camera_calibration.status()["active"]:
                raise RuntimeError("Finish or cancel checkerboard camera calibration first")
            if not self._calibration_workflow_ready:
                raise RuntimeError("Scientific calibration workflow is unavailable in this build.")
            if self.recorder.status().active:
                raise RuntimeError("Stop the current recording before starting calibration")
            status = self.status()
            imu = status.get("imu", {})
            if not imu.get("connected") or not imu.get("capture_active"):
                raise RuntimeError("Connect the IMU source before starting calibration")
            if not imu.get("thigh_ready") or not imu.get("shank_ready"):
                raise RuntimeError("Both thigh and shank IMUs must be READY")
            normalized_test_type = str(test_type or "").strip().lower()
            camera_calibration = status.get("camera_calibration", {})
            reference_validation = status.get("reference_validation", {})
            if camera_calibration.get("persistent_state") != "LOADED":
                raise RuntimeError(
                    "Complete the persistent checkerboard camera setup before IMU calibration"
                )
            if not reference_validation.get("persisted_passed"):
                raise RuntimeError(
                    "Complete and pass the one-time known-angle tag-reference validation before IMU calibration"
                )
            if normalized_test_type == "dynamic" and not status.get("tracking_valid"):
                raise RuntimeError(
                    "Tag reference must be VALID before the dynamic test starts. "
                    "Keep both tags visible in the validated camera setup."
                )
            if normalized_test_type == "dynamic":
                existing_run = Path(self.calibration.status().run_directory)
                self._assert_calibration_run_provenance_current(existing_run)

            with self._imu_filter_lock:
                base_parameters = self._imu_filter.parameters
            prepared = self.calibration.prepare_test(
                test_type,
                base_parameters=base_parameters,
            )
            with self._lock:
                tuning = self._tuning.to_dict()
                tuning["reference_backend"] = getattr(
                    self.reference, "description", type(self.reference).__name__
                )
                source_description = self.camera.description
            imu_description = self._format_imu_description(imu)
            tuning["imu_filter"] = self.imu_estimator_status()
            run_directory = Path(prepared["run_directory"])
            if normalized_test_type == "static":
                try:
                    self._write_calibration_run_provenance(
                        run_directory, base_parameters=base_parameters
                    )
                except Exception as exc:
                    self.calibration.fail(f"Could not freeze calibration provenance: {exc}")
                    raise
            try:
                self.recorder.request_start(
                    settings=tuning,
                    source=source_description,
                    imu_source=imu_description,
                    output_root=run_directory,
                    session_name=Path(str(prepared["archive_name"])).stem,
                    extra_metadata={
                        "recording_purpose": "marker_imu_filter_calibration",
                        "calibration_test_type": prepared["test_type"],
                        "automatic_duration_s": prepared["duration_s"],
                        "protocol_version": FINAL_PROTOCOL_VERSION,
                        "protocol": PROTOCOL_SUMMARY,
                        "runtime_provenance": self._provenance,
                        "reference_setup_sha256": self.reference_setup_store.current_sha256(),
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
            self._load_persistent_reference_setup()
            with self._new_frame:
                self._status = RuntimeStatus(
                    **{
                        **asdict(self._status),
                        "state": "CAMERA_PREVIEW",
                        "source": self.camera.description,
                        "tracking_reason": "initializing_tag_reference",
                        "error": "",
                    }
                )
                self._new_frame.notify_all()

            try:
                self.reference.open()
            except Exception as exc:
                self._reference_error = str(exc)
                self._reference_failed.set()
                with self._new_frame:
                    self._status = RuntimeStatus(
                        **{
                            **asdict(self._status),
                            "state": "REFERENCE_FAILED",
                            "tracking_valid": False,
                            "tracking_reason": "reference_backend_failed",
                            "error": self._reference_error,
                        }
                    )
                    self._new_frame.notify_all()
            else:
                self._reference_ready.set()

            capture_thread = threading.Thread(
                target=self._capture_loop,
                name="vision-imu-camera",
                daemon=True,
            )
            self._capture_thread = capture_thread
            capture_thread.start()
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
            self.reference.close()
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
        active_rotation: FrameRotation | None = None

        try:
            while not self._stop_event.is_set():
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                host_monotonic_ns = time.perf_counter_ns()
                with self._lock:
                    tuning = self._tuning

                if self._reset_tracking_event.is_set():
                    self._reset_tracking_event.clear()
                    active_rotation = None

                if active_rotation is not tuning.frame_rotation:
                    active_rotation = tuning.frame_rotation
                frame = self._rotate_frame(frame, tuning.frame_rotation, cv2)

                solved_camera = self.camera_calibration.process_frame(frame)
                if solved_camera is not None:
                    configure_camera = getattr(self.reference, "configure_camera", None)
                    if callable(configure_camera):
                        configure_camera(solved_camera)
                        self._persist_reference_setup()
                        self._reset_tracking_event.set()

                if self._reference_ready.is_set() and not self._reference_failed.is_set():
                    try:
                        reference_result = self.reference.process(frame)
                    except Exception as exc:
                        self._reference_error = str(exc)
                        self._reference_failed.set()
                        self._reference_ready.clear()
                        reference_result = MarkerReferenceResult(
                            angle_deg=None,
                            valid=False,
                            reason="reference_processing_failed",
                            quality=0.0,
                            thigh=None,
                            shank=None,
                        )
                elif self._reference_failed.is_set():
                    reference_result = MarkerReferenceResult(
                        angle_deg=None,
                        valid=False,
                        reason="reference_backend_failed",
                        quality=0.0,
                        thigh=None,
                        shank=None,
                    )
                else:
                    reference_result = MarkerReferenceResult(
                        angle_deg=None,
                        valid=False,
                        reason="initializing_tag_reference",
                        quality=0.0,
                        thigh=None,
                        shank=None,
                    )

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(1e-6, now - last_completed)
                last_completed = now
                if self._reference_ready.is_set() and not self._reference_failed.is_set():
                    ema_fps = (
                        instantaneous_fps
                        if ema_fps <= 0
                        else (0.9 * ema_fps + 0.1 * instantaneous_fps)
                    )
                else:
                    ema_fps = 0.0

                annotated = draw_marker_overlay(
                    frame,
                    reference_result,
                    processing_fps=ema_fps,
                    source_description=self.camera.description,
                    recording_active=self.recorder.status().active,
                )
                annotated = self.camera_calibration.draw_overlay(annotated)
                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
                if not ok:
                    continue

                frame_index += 1
                with self._imu_filter_lock:
                    imu_estimate = self._latest_imu_estimate
                difference_deg = None
                if (
                    reference_result.valid
                    and reference_result.angle_deg is not None
                    and imu_estimate.valid
                    and imu_estimate.knee_angle_deg is not None
                ):
                    difference_deg = float(reference_result.angle_deg) - float(
                        imu_estimate.knee_angle_deg
                    )

                thigh_center = (
                    reference_result.thigh.center if reference_result.thigh is not None else (None, None)
                )
                shank_center = (
                    reference_result.shank.center if reference_result.shank is not None else (None, None)
                )
                thigh_pose = reference_result.thigh_pose
                shank_pose = reference_result.shank_pose
                sample = {
                    "host_monotonic_ns": host_monotonic_ns,
                    "frame_index": frame_index,
                    "visual_knee_angle_deg": reference_result.angle_deg,
                    "tracking_valid": reference_result.valid,
                    "tracking_reason": reference_result.reason,
                    "marker_quality": reference_result.quality,
                    "thigh_tag_detected": reference_result.thigh_detected,
                    "shank_tag_detected": reference_result.shank_detected,
                    "thigh_tag_center_x": thigh_center[0],
                    "thigh_tag_center_y": thigh_center[1],
                    "shank_tag_center_x": shank_center[0],
                    "shank_tag_center_y": shank_center[1],
                    "thigh_pose_valid": bool(thigh_pose and thigh_pose.valid),
                    "shank_pose_valid": bool(shank_pose and shank_pose.valid),
                    "thigh_reprojection_rmse_px": (
                        thigh_pose.reprojection_rmse_px if thigh_pose is not None else None
                    ),
                    "shank_reprojection_rmse_px": (
                        shank_pose.reprojection_rmse_px if shank_pose is not None else None
                    ),
                    "thigh_view_angle_deg": (
                        thigh_pose.view_angle_deg if thigh_pose is not None else None
                    ),
                    "shank_view_angle_deg": (
                        shank_pose.view_angle_deg if shank_pose is not None else None
                    ),
                    "processing_fps": ema_fps,
                    "live_imu_filter_valid": imu_estimate.valid,
                    "live_imu_knee_angle_deg": imu_estimate.knee_angle_deg,
                    "live_imu_knee_speed_deg_s": imu_estimate.knee_speed_deg_s,
                    "live_angle_difference_deg": difference_deg,
                }
                with self._imu_filter_lock:
                    self._acquisition_history.append(
                        {
                            "host_monotonic_ns": host_monotonic_ns,
                            "visual_knee_angle_deg": reference_result.angle_deg,
                            "visual_valid": reference_result.valid,
                            "imu_knee_angle_deg": imu_estimate.knee_angle_deg,
                            "imu_valid": imu_estimate.valid,
                            "angle_difference_deg": difference_deg,
                        }
                    )
                self.recorder.write(annotated, preferred_fps=ema_fps or 15.0, sample=sample)

                if self._reference_failed.is_set():
                    state = "REFERENCE_FAILED"
                    error = self._reference_error
                elif self._reference_ready.is_set():
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
                        tracking_valid=reference_result.valid,
                        tracking_reason=reference_result.reason,
                        visual_knee_angle_deg=reference_result.angle_deg,
                        marker_quality=reference_result.quality,
                        thigh_tag_detected=reference_result.thigh_detected,
                        shank_tag_detected=reference_result.shank_detected,
                        processing_fps=ema_fps,
                        frame_index=frame_index,
                        error=error,
                        thigh_pose_valid=bool(thigh_pose and thigh_pose.valid),
                        shank_pose_valid=bool(shank_pose and shank_pose.valid),
                        thigh_reprojection_rmse_px=(
                            thigh_pose.reprojection_rmse_px if thigh_pose is not None else None
                        ),
                        shank_reprojection_rmse_px=(
                            shank_pose.reprojection_rmse_px if shank_pose is not None else None
                        ),
                        thigh_view_angle_deg=(
                            thigh_pose.view_angle_deg if thigh_pose is not None else None
                        ),
                        shank_view_angle_deg=(
                            shank_pose.view_angle_deg if shank_pose is not None else None
                        ),
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
    def _load_cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed") from exc
        return cv2
