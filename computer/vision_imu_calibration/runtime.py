from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .camera_source import OpenCvCameraSource
from .geometry import KneeAngleResult, calculate_knee_flexion_deg
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
        leg_side: LegSide = LegSide.AUTO,
        confidence_threshold: float = 0.30,
        recording_root: str | Path = "vision_calibration_recordings",
        imu_source: SerialImuSource | None = None,
        imu_source_factory: Any = SerialImuSource,
    ) -> None:
        self.camera = camera
        self.estimator = estimator
        self._tuning = RuntimeTuning(
            leg_side=leg_side,
            confidence_threshold=float(confidence_threshold),
        )
        self.recorder = AnnotatedVideoRecorder(recording_root)
        self._imu_source = imu_source
        self._imu_source_factory = imu_source_factory
        self._imu_lock = threading.RLock()
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

    @property
    def imu_source(self) -> SerialImuSource | None:
        with self._imu_lock:
            return self._imu_source

    def list_imu_ports(self) -> list[dict[str, str]]:
        return list_serial_ports()

    def connect_imu(self, port: str, *, baud_rate: int = IMUCAL_BAUD_RATE) -> dict[str, Any]:
        if self.recorder.status().active:
            raise RuntimeError("Stop recording before changing the IMU serial connection")
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
        new_source.start()
        if self._thread is not None and self._thread.is_alive():
            self._ensure_imu_thread()
        return new_source.status()

    def disconnect_imu(self) -> dict[str, Any]:
        if self.recorder.status().active:
            raise RuntimeError("Stop recording before disconnecting the IMU serial source")
        with self._imu_lock:
            source, self._imu_source = self._imu_source, None
        if source is not None:
            source.stop()
        return _empty_imu_status()

    def camera_status(self) -> dict[str, Any]:
        profile = self._camera_profile_values()
        return {
            "profile": self._camera_profile,
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
        payload["recording"] = asdict(self.recorder.status())
        source = self.imu_source
        payload["imu"] = source.status() if source is not None else _empty_imu_status()
        payload["camera"] = self.camera_status()
        return payload

    def tuning(self) -> dict[str, Any]:
        with self._lock:
            return self._tuning.to_dict()

    def update_tuning(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._tuning = RuntimeTuning.from_mapping(payload, current=self._tuning)
            self._status = RuntimeStatus(
                **{
                    **asdict(self._status),
                    "requested_leg": self._tuning.leg_side.value,
                }
            )
            return self._tuning.to_dict()

    def start_recording(self) -> dict[str, Any]:
        with self._lock:
            tuning = self._tuning.to_dict()
            tuning["pose_backend"] = getattr(self.estimator, "description", type(self.estimator).__name__)
            source = self.camera.description
            imu_description = ""
            imu_source = self.imu_source
            if imu_source is not None:
                imu_status = imu_source.status()
                imu_description = f"{imu_status['port']}@{imu_status['baud_rate']}"
        self.recorder.request_start(
            settings=tuning,
            source=source,
            imu_source=imu_description,
        )
        if imu_source is not None and imu_source.configured:
            try:
                imu_source.start_capture()
            except Exception:
                self.recorder.stop()
                raise
        return asdict(self.recorder.status())

    def stop_recording(self) -> dict[str, Any]:
        source = self.imu_source
        if source is not None and source.status()["capture_active"]:
            try:
                source.stop_capture()
            except Exception:
                pass
        return asdict(self.recorder.stop())

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
            if sample is not None:
                self.recorder.write_imu(sample)

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
                        raw_angle = calculate_knee_flexion_deg(
                            raw_leg,
                            confidence_threshold=tuning.confidence_threshold,
                            min_segment_length=tuning.min_segment_length,
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
                }
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
