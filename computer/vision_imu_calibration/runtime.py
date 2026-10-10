from __future__ import annotations

import threading
import time
import json
import math
import os
import uuid
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
from .positioning import PositioningCheck, visibility_reason
from .reference_validation import (GONIOMETER_TARGETS, guided_capture_report,
                                   guided_repeatability, measured_reference)
from .pose_estimator import PoseEstimator
from .recorder import AnnotatedVideoRecorder
from .tuning import FrameRotation, RuntimeTuning


CAMERA_PROFILES: dict[str, tuple[int, int, float]] = {
    "fast": (640, 480, 30.0),
    "quality": (1280, 720, 30.0),
}
POSE_PRESET_SCHEMA_VERSION = 1
POSE_PRESET_FILENAME = "pose_tuning_preset_v1.json"
POSE_PRESET_FIELDS = (
    "confidence_threshold",
    "landmark_smoothing",
    "angle_smoothing",
    "min_segment_length",
    "max_angle_jump_deg",
)


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
        self._pose_preset_path = self.calibration_root / POSE_PRESET_FILENAME
        self._pose_preset_error = ""
        self._pose_preset_loaded = False
        self._saved_pose_tuning: dict[str, float] | None = None
        self._load_pose_preset()
        self.recorder = AnnotatedVideoRecorder(self.recording_root)
        self.calibration = CalibrationWorkflow(self.calibration_root)
        restored_setup = self.calibration.setup_metadata()
        self._session_setup_id = str((restored_setup or {}).get("description", "")).strip()
        if (restored_setup is not None
                and restored_setup.get("pose_tuning") != self._tuning.to_dict()):
            self.calibration.invalidate_setup(
                "Saved calibration pose configuration differs from this restart; start a new calibration"
            )
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
        self._overlay_stroke_scale = 1.0  # Session-only visual preference; never affects pose data.
        self._pose_error = ""
        self._reset_tracking_event = threading.Event()
        self._camera_profile = self._infer_camera_profile()
        self._positioning = PositioningCheck()
        self._reference_root = self.recording_root / "goniometer"
        self._reference_recorder = AnnotatedVideoRecorder(self._reference_root)
        self._reference = {"state": "IDLE", "message": "Optional laboratory validation",
                           "hide_angle": False, "result": None, "reused_capture_id": ""}
        self._reference_inputs: dict[str, Any] = {}
        self._guided_reference: dict[str, Any] | None = None
        self._landmark_scores: dict[str, float | None] = {"hip": None, "knee": None, "ankle": None}
        self._pose_tuning_history: deque[dict[str, Any]] = deque(maxlen=1200)
        self._reference_samples: list[dict[str, Any]] = []
        self._reference_cancel = threading.Event()
        self._reference_thread: threading.Thread | None = None
        self._reference_captures: list[dict[str, Any]] = []
        self._reference_load_errors: list[str] = []
        # Only small reports are loaded; sample arrays remain in the exported file.
        if self._reference_root.is_dir():
            for report_path in sorted(self._reference_root.glob("goniometer_*.json")):
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if report.get("status") == "Measured":
                        capture_id = str(report["capture_id"])
                        if (uuid.UUID(hex=capture_id).hex != capture_id
                                or report_path.stem != "goniometer_" + capture_id
                                or report["recording_zip"] != report_path.with_suffix(".zip").name):
                            raise ValueError("Saved capture identity / archive name mismatch")
                        measured_reference(report["reference_angle_deg"])
                        if not isinstance(report["profile"], dict) or not isinstance(report["setup"], str):
                            raise ValueError("Saved setup metadata missing")
                        if not report["hold"]["accepted"] or not report["metrics"]["sufficient_data"]:
                            raise ValueError("Saved capture does not have accepted hold data")
                        for key in ("raw", "filtered"):
                            for metric in ("mean_deg", "bias_deg"):
                                if not math.isfinite(report["metrics"][key][metric]):
                                    raise ValueError("Saved capture has nonfinite statistics")
                        report.pop("samples", None)
                        self._reference_captures.append(report)
                except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
                    self._reference_load_errors.append(f"{report_path.name}: {exc}")
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
        return status.active or status.state.startswith("ANALYZING") or self._reference_busy()

    def _reference_busy(self) -> bool:
        with self._lock:
            return self._reference["state"] in {"CAPTURING", "FINALIZING"}

    def _vision_setup(self) -> dict[str, Any]:
        return {"camera": self.camera_status(), "tuning": self._tuning.to_dict(),
                "pose_backend": getattr(self.estimator, "description", type(self.estimator).__name__),
                "tracked_leg": self._status.tracked_leg}

    @staticmethod
    def _pose_tuning_values(tuning: RuntimeTuning) -> dict[str, float]:
        payload = tuning.to_dict()
        return {name: float(payload[name]) for name in POSE_PRESET_FIELDS}

    def _load_pose_preset(self) -> None:
        if not self._pose_preset_path.is_file():
            return
        try:
            payload = json.loads(self._pose_preset_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != POSE_PRESET_SCHEMA_VERSION:
                raise ValueError("unsupported pose preset schema")
            values = payload.get("pose_tuning")
            if not isinstance(values, Mapping) or set(values) != set(POSE_PRESET_FIELDS):
                raise ValueError("pose preset must contain exactly the five supported settings")
            updated = RuntimeTuning.from_mapping(values, current=self._tuning)
            self._tuning = updated
            self._saved_pose_tuning = self._pose_tuning_values(updated)
            self._pose_preset_loaded = True
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._pose_preset_error = f"Could not load saved pose settings: {exc}"

    def pose_preset_status(self) -> dict[str, Any]:
        with self._lock:
            current = self._pose_tuning_values(self._tuning)
            saved = dict(self._saved_pose_tuning) if self._saved_pose_tuning is not None else None
            error = self._pose_preset_error
            loaded = self._pose_preset_loaded
        return {
            "schema_version": POSE_PRESET_SCHEMA_VERSION,
            "path": str(self._pose_preset_path),
            "exists": self._pose_preset_path.is_file(),
            "loaded": loaded,
            "saved": saved is not None and saved == current,
            "error": error,
        }

    def save_tuning_preset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("Expected a JSON object")
        if not set(POSE_PRESET_FIELDS).issubset(payload):
            raise ValueError("All five pose settings are required")
        self.update_tuning({name: payload[name] for name in POSE_PRESET_FIELDS})
        with self._lock:
            values = self._pose_tuning_values(self._tuning)
        document = {
            "schema_version": POSE_PRESET_SCHEMA_VERSION,
            "pose_tuning": values,
        }
        temporary = self._pose_preset_path.with_name(
            self._pose_preset_path.name + ".tmp-" + uuid.uuid4().hex
        )
        try:
            temporary.write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self._pose_preset_path)
        except OSError as exc:
            self._pose_preset_error = f"Could not save pose settings: {exc}"
            raise RuntimeError(self._pose_preset_error) from exc
        finally:
            temporary.unlink(missing_ok=True)
        with self._lock:
            self._saved_pose_tuning = values
            self._pose_preset_loaded = True
            self._pose_preset_error = ""
        return self.pose_preset_status()

    def _session_description(self, explicit: str = "", *, new_run: bool = False) -> str:
        text = str(explicit or "").strip()
        if len(text) > 1000:
            raise ValueError("Calibration setup description must be at most 1000 characters")
        with self._lock:
            if new_run:
                self._session_setup_id = text or ("session_" + uuid.uuid4().hex)
            elif text:
                return text
            elif not self._session_setup_id:
                restored = self.calibration.setup_metadata()
                self._session_setup_id = str((restored or {}).get("description", "")).strip()
            if not self._session_setup_id:
                raise RuntimeError("Calibration session setup is missing; start a new static calibration")
            return self._session_setup_id

    def _calibration_setup_snapshot(
        self,
        description: str,
        imu_status: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        text = str(description or "").strip()
        if not text:
            return None
        if len(text) > 1000:
            raise ValueError("Calibration setup description must be at most 1000 characters")
        status = dict(imu_status or (self.imu_source.status() if self.imu_source is not None else {}))
        with self._lock:
            tuning = self._tuning.to_dict()
            backend = getattr(self.estimator, "description", type(self.estimator).__name__)
            tracked_leg = self._status.tracked_leg
            camera_source = self.camera.description
            camera_profile = self._camera_profile
        with self._imu_filter_lock:
            filter_parameters = self._imu_filter.parameters.to_dict()
            filter_parameters.pop("rotation_axis_name", None)
            filter_parameters["rotation_axis"] = ("x", "y", "z")[self._imu_filter.parameters.rotation_axis]
        return {
            "description": text,
            "camera_profile": camera_profile,
            "camera_source": camera_source,
            "pose_backend": backend,
            "pose_tuning": tuning,
            "tracked_leg": tracked_leg,
            "imu_source": self._format_imu_description(status),
            "imu_transport": str(status.get("transport", "") or ""),
            "filter_parameters": filter_parameters,
        }

    def positioning_action(self, action: str, manual_side_on: bool = False) -> dict[str, Any]:
        with self._lock:
            if action == "show":
                self._positioning.enabled = True
            elif action in {"check", "start"}:
                if self._status.state != "RUNNING":
                    raise RuntimeError("Wait for live pose tracking")
                self._positioning.begin(manual_side_on)
            elif action == "finish":
                return self._positioning.finish(time.perf_counter())
            else:
                raise ValueError("Positioning action must be check, show, start, or finish")
            return self._positioning.snapshot()

    def _guided_snapshot(self) -> dict[str, Any] | None:
        session = self._guided_reference
        if session is None:
            return None
        index = session["index"]
        targets = session["targets"]
        captures = session["captures"]
        return {"session_id": session["id"], "index": index, "total": len(targets),
                "target_deg": targets[index] if index < len(targets) else None,
                "repeat": (index % 2) + 1 if index < len(targets) else None,
                "captures": list(captures), "complete": index == len(targets),
                "report_url": session.get("report_url", "")}

    def _prepare_guided_point(self) -> None:
        session = self._guided_reference
        if session is None or session["index"] >= len(session["targets"]):
            raise RuntimeError("No pending guided goniometer position")
        index = session["index"]
        target = session["targets"][index]
        self._reference_inputs = {
            "capture_id": uuid.uuid4().hex,
            "target_deg": target,
            "reference_angle_deg": float(target),
            "setup": "guided:" + session["id"],
            "profile": session["profile"],
            "reference_method": "operator_positioned_with_physical_goniometer_at_requested_reference",
            "session_id": session["id"],
            "repeat_index": (index % 2) + 1,
        }
        self._reference.update(state="ENTRY", hide_angle=True, result=None,
                               reused_capture_id="",
                               message=f"Position leg at {target}° using the physical goniometer, then Capture")
        self._latest_jpeg = None

    def overlay_stroke_status(self) -> dict[str, float]:
        with self._lock:
            return {"stroke_scale": self._overlay_stroke_scale}

    def set_overlay_stroke_scale(self, value: float) -> dict[str, float]:
        """Change only live overlay stroke width; keep pose geometry and markers unchanged."""
        if isinstance(value, bool):
            raise ValueError("Overlay stroke scale must be numeric")
        try:
            scale = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Overlay stroke scale must be numeric") from exc
        if not math.isfinite(scale) or not 0.4 <= scale <= 1.2:
            raise ValueError("Overlay stroke scale must be between 0.4 and 1.2")
        with self._lock:
            self._overlay_stroke_scale = scale
        return {"stroke_scale": scale}

    def reference_status(self) -> dict[str, Any]:
        with self._lock:
            compatible = [r for r in self._reference_captures
                          if r.get("setup") == self._reference_inputs.get("setup")
                          and r.get("profile") == self._vision_setup()]
            return {**self._reference, "guided": self._guided_snapshot(), "captures": list(self._reference_captures),
                    "load_errors": list(self._reference_load_errors),
                    "repeatability": guided_repeatability(compatible),
                    "repeatability_scope": "current described setup and exact camera/pose profile"}

    def reference_action(self, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._calibration_action_lock, self._lock:
            if self._reference_busy():
                raise RuntimeError("Wait for the current goniometer hold to finish")
            if action == "start_session":
                if self.recorder.status().active or self._calibration_locked():
                    raise RuntimeError("Finish recording or calibration before guided validation")
                self._guided_reference = {
                    "id": uuid.uuid4().hex,
                    "targets": tuple(v for target in GONIOMETER_TARGETS for v in (target, target)),
                    "index": 0, "captures": [], "profile": self._vision_setup(),
                }
                self._prepare_guided_point()
            elif action == "next":
                if self._guided_reference is None:
                    raise RuntimeError("Start the guided static check first")
                if self._reference["state"] not in {"MEASURED", "RETRY"}:
                    raise RuntimeError("Finish or retry the current position first")
                if self._guided_reference["index"] >= len(self._guided_reference["targets"]):
                    raise RuntimeError("All ten measurements are complete")
                if self._guided_reference["profile"] != self._vision_setup():
                    raise RuntimeError("Camera/pose profile changed; start a new session")
                self._prepare_guided_point()
            elif action == "cancel":
                self._guided_reference = None
                self._reference.update(state="IDLE", hide_angle=False, message="Validation closed")
            elif action == "begin":
                self._guided_reference = None
                if self.recorder.status().active or self._calibration_locked():
                    raise RuntimeError("Finish the current recording/calibration first")
                target = payload.get("target_deg")
                if isinstance(target, bool) or target not in (*GONIOMETER_TARGETS, 150):
                    raise ValueError("Choose a target prompt: 90, 120, 140, 150, 160, or 175")
                setup = str(payload.get("setup", "")).strip()
                if not setup or len(setup) > 2000:
                    raise ValueError("Describe camera mounting/distance, floor mark, subject, and goniometer")
                self._reference_inputs = {"capture_id": uuid.uuid4().hex, "target_deg": target,
                                          "setup": setup, "profile": self._vision_setup(),
                                          "reference_method": "operator_measured_goniometer"}
                self._reference.update(state="ENTRY", hide_angle=True, result=None,
                                       reused_capture_id="", message=f"Set ~{target} deg; enter the actual goniometer measurement")
                # A previously published JPEG contains a numerical angle. Do not
                # serve it while waiting for the first blinded camera frame.
                self._latest_jpeg = None
            elif action == "confirm":
                if self._reference["state"] != "ENTRY":
                    raise RuntimeError("Begin a new point before entering its measurement")
                self._reference_inputs["reference_angle_deg"] = measured_reference(payload.get("reference_angle_deg"))
                self._reference.update(state="CONFIRMED", hide_angle=False,
                                       message="Manual value confirmed; keep still and Capture a 3-second hold")
            elif action == "capture":
                guided = self._guided_reference is not None
                allowed = {"ENTRY", "CONFIRMED"} if guided else {"CONFIRMED"}
                if self._reference["state"] not in allowed:
                    raise RuntimeError("Prepare a guided point or confirm a measured value before Capture")
                if self.recorder.status().active or self._calibration_locked():
                    raise RuntimeError("Finish the current recording/calibration first")
                if self._reference_inputs["profile"] != self._vision_setup():
                    raise RuntimeError("Camera/pose profile changed; begin the point again")
                if not self._status.tracking_valid or self._status.state != "RUNNING":
                    raise RuntimeError("Live pose tracking must be valid")
                self._reference_samples = []
                self._reference_inputs["hold_start_host_monotonic_ns"] = time.perf_counter_ns()
                self._reference_cancel = threading.Event()
                self._reference_recorder.request_start(
                    settings={**self._reference_inputs["profile"]["tuning"],
                              "pose_backend": self._reference_inputs["profile"]["pose_backend"]},
                    source=self.camera.description,
                    session_name="goniometer_" + self._reference_inputs["capture_id"],
                    extra_metadata={"recording_purpose": "goniometer_static_vision_validation",
                                    "goniometer_reference": dict(self._reference_inputs),
                                    "angle_convention": "180 deg straight; smaller means increased flexion"})
                self._reference.update(state="CAPTURING", message="Hold still for 3 seconds")
                self._reference_thread = threading.Thread(target=self._finish_reference_hold,
                                                         name="goniometer-hold", daemon=True)
                self._reference_thread.start()
            elif action == "reuse":
                if payload.get("unchanged_setup") is not True:
                    raise ValueError("Operator must confirm the physical setup is unchanged")
                capture = next((r for r in self._reference_captures
                                if r["capture_id"] == payload.get("capture_id")), None)
                if capture is None:
                    raise ValueError("Choose an accepted saved capture")
                if capture["profile"] != self._vision_setup() or capture["setup"] != str(payload.get("setup", "")).strip():
                    raise ValueError("Saved setup/profile differs; repeat laboratory validation")
                archive = self._reference_root / capture["recording_zip"]
                from .reference_validation import _digest
                if not archive.is_file() or _digest(archive) != capture["metrics"]["recording_sha256"]:
                    raise ValueError("Saved recording missing or changed; repeat validation")
                self._reference_inputs = {"setup": capture["setup"], "profile": capture["profile"]}
                self._reference.update(state="REUSED", hide_angle=False,
                                       reused_capture_id=capture["capture_id"], result=capture,
                                       message="Saved comparison selected; operator confirms equivalent physical setup")
            else:
                raise ValueError("Unknown goniometer action")
            return self.reference_status()

    def _finish_reference_hold(self) -> None:
        cancelled = self._reference_cancel.wait(3.0)
        with self._lock:
            self._reference.update(state="FINALIZING", message="Finalizing reference ZIP/report")
            inputs = dict(self._reference_inputs)
            inputs["hold_end_host_monotonic_ns"] = time.perf_counter_ns()
            samples = list(self._reference_samples)
        try:
            recording = self._reference_recorder.stop()
            if cancelled:
                raise RuntimeError("Capture interrupted by runtime stop; Needs retry")
            if recording.error:
                raise RuntimeError(recording.error)
            report = guided_capture_report(Path(recording.archive_path), inputs, samples)
            with self._lock:
                compatible = [r for r in self._reference_captures
                              if r.get("setup") == inputs["setup"] and r.get("profile") == inputs["profile"]]
            report["repeatability"] = guided_repeatability(compatible + [report])
            report["repeatability_scope"] = "same described physical setup and camera/pose profile"
            report_path = Path(recording.archive_path).with_suffix(".json")
            report["report_url"] = "/exports/recordings/goniometer/" + report_path.name
            report["download_url"] = "/exports/recordings/goniometer/" + Path(recording.archive_path).name
            with report_path.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
            report.pop("samples", None)
            with self._lock:
                if report["status"] == "Measured":
                    self._reference_captures.append(report)
                guided = self._guided_reference
                if guided is not None and inputs.get("session_id") == guided["id"] and report["status"] == "Measured":
                    guided["captures"].append({
                        "target_deg": inputs["target_deg"], "repeat_index": inputs["repeat_index"],
                        "raw_mean_deg": report["metrics"]["raw"]["mean_deg"],
                        "raw_error_deg": report["metrics"]["raw"]["bias_deg"],
                        "filtered_mean_deg": report["metrics"]["filtered"]["mean_deg"],
                        "filtered_error_deg": report["metrics"]["filtered"]["bias_deg"],
                        "capture_id": inputs["capture_id"], "report_url": report["report_url"]})
                    guided["index"] += 1
                    if guided["index"] == len(guided["targets"]):
                        import statistics
                        errors = [c["raw_error_deg"] for c in guided["captures"]]
                        summary = {"schema_version": 1, "session_id": guided["id"],
                                   "camera_pose_profile": guided["profile"],
                                   "protocol_targets_deg": list(GONIOMETER_TARGETS),
                                   "independent_repositions_per_target": 2,
                                   "captured_positions": list(guided["captures"]),
                                   "raw_mae_deg": statistics.mean(abs(e) for e in errors),
                                   "raw_rmse_deg": (statistics.mean(e*e for e in errors)) ** .5,
                                   "method": "operator-positioned static goniometer references; no interpolation or correction",
                                   "limitations": "2-D camera geometry and physical goniometer placement uncertainty"}
                        name = "static_validation_session_" + guided["id"] + ".json"
                        (self._reference_root / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
                        guided["report_url"] = "/exports/recordings/goniometer/" + name
                state = "COMPLETE" if guided and guided["index"] == len(guided["targets"]) else ("MEASURED" if report["status"] == "Measured" else "RETRY")
                self._reference.update(state=state, hide_angle=False, result=report,
                                       message=report["hold"]["reason"] or "Capture accepted; move away before next placement")
                self._latest_jpeg = None
        except Exception as exc:
            with self._lock:
                self._reference.update(state="RETRY", message=str(exc), result=None)

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
        if len(rows) > max_points:
            step = max(1, len(rows) // max_points)
            rows = rows[::step][-max_points:]
        with self._lock:
            if self._reference["hide_angle"]:
                return [{**row, "visual_knee_angle_deg": None, "visual_valid": False,
                         "angle_difference_deg": None} for row in rows]
        return rows

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
        with self._lock:
            self._positioning.reset("Camera profile changed; repeat positioning")
            if self._reference["reused_capture_id"]:
                self._reference.update(state="IDLE", result=None, message="Camera profile changed; saved selection reset")
            self._reference["reused_capture_id"] = ""
        self.calibration.invalidate_setup(
            "Camera profile changed; start a new calibration before held-out validation"
        )
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
            self._positioning.reset("Tracking restarted; repeat positioning")
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
        self._reference_cancel.set()
        timer_cancel = self._calibration_timer_cancel
        if timer_cancel is not None:
            timer_cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))
        self.recorder.close()
        reference_thread = self._reference_thread
        if reference_thread is not None and reference_thread.is_alive():
            reference_thread.join(timeout=max(0.0, float(timeout_s)))
        with self._lock:
            self._positioning.reset("Tracking stopped; repeat positioning")
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
            payload["positioning"] = self._positioning.snapshot()
            if self._reference["hide_angle"]:
                payload["raw_knee_angle_deg"] = None
                payload["visual_knee_angle_deg"] = None
        with self._lock:
            payload["landmark_confidences"] = dict(self._landmark_scores)
        payload["goniometer"] = self.reference_status()
        recording_payload = asdict(self.recorder.status())
        recording_payload.update(self._recording_export_payload(recording_payload))
        payload["recording"] = recording_payload
        source = self.imu_source
        payload["imu"] = source.status() if source is not None else _empty_imu_status()
        payload["imu_estimator"] = self.imu_estimator_status()
        payload["camera"] = self.camera_status()
        calibration_payload = asdict(self.calibration.status())
        calibration_payload.update(self._calibration_export_payload(calibration_payload))
        calibration_payload["session_setup_id"] = self._session_setup_id
        heldout = self.calibration.heldout_status()
        calibration_payload["heldout_validation"] = heldout
        run_directory = _optional_owned_path(
            calibration_payload.get("run_directory"), self.calibration_root, directory=True
        )
        validation_path = run_directory / "heldout_validation.json" if run_directory else None
        if validation_path is not None and validation_path.is_file():
            relative = validation_path.relative_to(self.calibration_root)
            calibration_payload["heldout_validation_url"] = (
                f"/exports/calibration/{quote(relative.as_posix())}"
            )
            calibration_payload["heldout_validation_display_path"] = (
                f"{self.calibration_root.name}/{relative.as_posix()}"
            )
        else:
            calibration_payload["heldout_validation_url"] = ""
            calibration_payload["heldout_validation_display_path"] = ""
        payload["calibration"] = calibration_payload
        payload["pose_preset"] = self.pose_preset_status()
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

    def pose_tuning_history(self, seconds: float = 10.0) -> dict[str, Any]:
        if not 0 < seconds <= 30:
            raise ValueError("History window must be within 0..30 seconds")
        cutoff = time.perf_counter_ns() - int(seconds * 1e9)
        with self._lock:
            rows = [dict(row) for row in self._pose_tuning_history
                    if row["host_monotonic_ns"] >= cutoff]
            if self._reference["hide_angle"]:
                for row in rows:
                    row["raw_angle_deg"] = None
                    row["filtered_angle_deg"] = None
            return {"rows": rows, "tuning": self._tuning.to_dict()}

    def tuning(self) -> dict[str, Any]:
        with self._lock:
            return self._tuning.to_dict()

    def update_tuning(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._calibration_action_lock:
            if self._calibration_locked():
                raise RuntimeError("Stop the active calibration/validation before changing pose settings")
            if self.recorder.status().active:
                raise RuntimeError("Stop recording before changing pose settings")
            with self._lock:
                updated = RuntimeTuning.from_mapping(payload, current=self._tuning)
                changed = updated != self._tuning
                if changed:
                    self._positioning.reset("Pose tuning changed; repeat tracking check")
                    if self._reference["reused_capture_id"]:
                        self._reference.update(state="IDLE", result=None, message="Pose tuning changed; saved selection reset")
                    self._reference["reused_capture_id"] = ""
                self._tuning = updated
                self._status = RuntimeStatus(
                    **{
                        **asdict(self._status),
                        "requested_leg": self._tuning.leg_side.value,
                    }
                )
                result = self._tuning.to_dict()
            if changed:
                self.calibration.invalidate_setup(
                    "Pose configuration changed; start a new calibration before held-out validation"
                )
            return result

    @staticmethod
    def _format_imu_description(status: Mapping[str, Any]) -> str:
        port = str(status.get("port", "") or "")
        baud_rate = int(status.get("baud_rate", 0) or 0)
        return f"{port}@{baud_rate}" if port and baud_rate > 0 else port

    def start_recording(
        self,
        setup: str = "",
        *,
        heldout: bool | None = None,
    ) -> dict[str, Any]:
        if self._calibration_locked():
            raise RuntimeError("A calibration test is already recording")
        runtime_status = self.status()
        calibration_status = runtime_status.get("calibration", {})
        heldout_context: dict[str, Any] | None = None
        heldout_setup: dict[str, Any] | None = None
        validation_id = ""
        calibration_complete = (
            calibration_status.get("valid") and calibration_status.get("state") == "COMPLETE"
        )
        use_heldout = heldout is True or (
            heldout is None and calibration_complete and bool(str(setup).strip())
        )
        if heldout is True and not calibration_complete:
            raise RuntimeError("Complete a valid Bayesian calibration before recording held-out data")
        if use_heldout:
            imu = runtime_status.get("imu", {})
            if (not imu.get("connected") or not imu.get("capture_active")
                    or not imu.get("thigh_ready") or not imu.get("shank_ready")
                    or not runtime_status.get("tracking_valid")):
                raise RuntimeError("Held-out recording requires READY IMUs and VALID visual tracking")
            heldout_context = self.calibration.heldout_context()
            description = self._session_description(setup)
            heldout_setup = self._calibration_setup_snapshot(description, imu)
            if heldout_setup != heldout_context["calibration_setup"]:
                raise RuntimeError("Physical/camera/filter setup changed; repeat calibration or restore it")
            validation_id = uuid.uuid4().hex
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
        tuning["pose_preset"] = self.pose_preset_status()
        extra_metadata: dict[str, Any] | None = None
        session_name: str | None = None
        if heldout_context is not None and heldout_setup is not None:
            session_name = "heldout_" + validation_id
            extra_metadata = {
                "recording_purpose": "vision_imu_filter_heldout_validation",
                "validation_id": validation_id,
                "calibration_run_id": calibration_status["run_id"],
                "calibration_report_sha256": heldout_context["calibration_report_sha256"],
                "training_archive_sha256": heldout_context["training_sha256"],
                "parameter_snapshot_sha256": heldout_context["parameter_sha256"],
                "fixed_lag_ms": 0.0,
                "pairing_policy": heldout_context["calibration_report"]["optimization"]["reference_sampling_policy"],
                "calibration_setup": heldout_setup,
            }
        self.recorder.request_start(
            settings=tuning,
            source=source,
            imu_source=imu_description,
            session_name=session_name,
            extra_metadata=extra_metadata,
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

    def start_calibration_test(self, test_type: str, setup: str = "") -> dict[str, Any]:
        with self._calibration_action_lock:
            if self._reference_busy():
                raise RuntimeError("Finish the goniometer hold before starting calibration")
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

            normalized_test = str(test_type or "").strip().lower()
            if normalized_test not in {"static", "dynamic"}:
                raise ValueError("Calibration test_type must be 'static' or 'dynamic'")
            description = self._session_description(
                setup,
                new_run=normalized_test == "static",
            )
            with self._imu_filter_lock:
                base_parameters = self._imu_filter.parameters
            setup_metadata = self._calibration_setup_snapshot(description, imu)
            prepared = self.calibration.prepare_test(
                normalized_test,
                base_parameters=base_parameters,
                setup_metadata=setup_metadata,
            )
            with self._lock:
                tuning = self._tuning.to_dict()
                tuning["pose_backend"] = getattr(
                    self.estimator, "description", type(self.estimator).__name__
                )
                source_description = self.camera.description
            imu_description = self._format_imu_description(imu)
            tuning["imu_filter"] = self.imu_estimator_status()
            tuning["pose_preset"] = self.pose_preset_status()
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
                        "calibration_setup": setup_metadata,
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

    def validate_latest_recording(
        self,
        *,
        setup: str = "",
        unchanged_setup: bool,
    ) -> dict[str, Any]:
        with self._calibration_action_lock:
            if self._reference_busy() or self.calibration.status().active:
                raise RuntimeError("Finish the current acquisition before held-out validation")
            recording = self.recorder.status()
            if recording.active:
                raise RuntimeError("Stop and finalize the new recording before validation")
            archive = _optional_owned_path(recording.archive_path, self.recording_root)
            if archive is None:
                raise RuntimeError("No finalized recording is available for validation")
            imu_status = self.imu_source.status() if self.imu_source is not None else {}
            description = self._session_description(setup)
            current_setup = self._calibration_setup_snapshot(description, imu_status)
            return self.calibration.validate_heldout(
                archive,
                current_setup=current_setup,
                unchanged_setup=bool(unchanged_setup),
            )

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
                    with self._lock:
                        self._positioning.reset("Tracking reset; repeat positioning")
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
                raw_leg = None
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

                with self._lock:
                    self._landmark_scores = {
                        key: (getattr(raw_leg, key).confidence if raw_leg is not None else None)
                        for key in ("hip", "knee", "ankle")
                    }
                    # Raw observations detect real movement; display smoothing
                    # must not conceal off-frame points or transient dropouts.
                    previous_side = self._positioning.context[4] if self._positioning.context else None
                    side = raw_leg.side.value if raw_leg is not None else previous_side
                    context = (self.camera.description, self._camera_profile,
                               tuning.frame_rotation.value, tuning.leg_side.value, side,
                               frame.shape[1], frame.shape[0])
                    self._positioning.set_context(context)
                    if (self._reference["reused_capture_id"] and
                            self._reference_inputs.get("profile", {}).get("tracked_leg") != side):
                        self._reference.update(state="IDLE", result=None, reused_capture_id="",
                                               message="Tracked leg changed; saved comparison selection reset")
                    positioning_angle = raw_angle if filtered_angle.valid else filtered_angle
                    self._positioning.observe(host_monotonic_ns / 1e9, raw_leg,
                                              positioning_angle, tuning.confidence_threshold)
                    position_status = self._positioning.snapshot()
                    hide_angle = self._reference["hide_angle"]
                    stroke_scale = self._overlay_stroke_scale
                annotated = draw_leg_overlay(
                    frame,
                    leg,
                    filtered_angle,
                    inference_fps=ema_fps,
                    source_description=self.camera.description,
                    recording_active=self.recorder.status().active,
                    hide_angle=hide_angle,
                    positioning=position_status,
                    stroke_scale=stroke_scale,
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
                with self._lock:
                    self._pose_tuning_history.append({
                        "frame_index": frame_index,
                        "host_monotonic_ns": host_monotonic_ns,
                        "raw_angle_deg": raw_angle.angle_deg,
                        "filtered_angle_deg": filtered_angle.angle_deg,
                        "tracking_valid": filtered_angle.valid,
                        "tracking_reason": filtered_angle.reason,
                        "hip": self._landmark_scores["hip"],
                        "knee": self._landmark_scores["knee"],
                        "ankle": self._landmark_scores["ankle"],
                        "tracked_leg": tracked_leg,
                    })
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
                with self._lock:
                    if (self._reference["state"] == "CAPTURING" and host_monotonic_ns
                            >= self._reference_inputs["hold_start_host_monotonic_ns"]):
                        observation = {"host_monotonic_ns": host_monotonic_ns,
                                       "tracked_leg": tracked_leg,
                                       "pose_confidence": raw_angle.confidence,
                                       "raw_tracking_valid": raw_angle.valid,
                                       "tracking_valid": filtered_angle.valid,
                                       "tracking_reason": filtered_angle.reason,
                                       "raw_knee_angle_deg": raw_angle.angle_deg,
                                       "visual_knee_angle_deg": filtered_angle.angle_deg,
                                       "valid": filtered_angle.valid and not visibility_reason(
                                           raw_leg, raw_angle, tuning.confidence_threshold)}
                        if (self._reference_inputs["profile"]["tuning"] != tuning.to_dict()
                                or self._reference_inputs["profile"] != self._vision_setup()):
                            observation["valid"] = False
                        for name in ("hip", "knee", "ankle"):
                            point = getattr(raw_leg, name) if raw_leg is not None else None
                            for axis in ("x", "y"):
                                observation[f"{name}_{axis}"] = getattr(point, axis) if point is not None else None
                        for key, value in observation.items():
                            if isinstance(value, float) and not math.isfinite(value):
                                observation[key] = None
                                observation["valid"] = False
                        self._reference_recorder.write(annotated, preferred_fps=ema_fps or 15.0, sample=sample)
                        self._reference_samples.append(observation)

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
                    if hide_angle == self._reference["hide_angle"]:
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
