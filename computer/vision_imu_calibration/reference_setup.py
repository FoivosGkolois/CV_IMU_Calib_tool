from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .camera_calibration import (
    MAX_CALIBRATION_RMS_PX,
    CameraCalibrationSolveResult,
    CheckerboardSpecification,
)
from .marker_reference import APRILTAG_DICTIONARY_NAME, CameraCalibration


REFERENCE_SETUP_SCHEMA_VERSION = 1
REFERENCE_SETUP_FILE = "reference_setup.yaml"
REFERENCE_SETUP_LOG_FILE = "REFERENCE_SETUP_LOG.txt"


@dataclass(frozen=True)
class ReferenceSetupLoadResult:
    calibration: CameraCalibration | None
    state: str
    message: str
    saved_utc: str = ""
    camera_description: str = ""
    rms_reprojection_error_px: float | None = None

    @property
    def valid(self) -> bool:
        return self.calibration is not None and self.state == "LOADED"


class ReferenceSetupStore:
    """Persistent owner for the validated camera/reference setup.

    On Arduino App Lab this root is placed under the reserved app ``data``
    directory, which is the platform-defined persistent application-data area.
    On PC runs the caller supplies the project ``reference_setup`` directory.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.yaml_path = self.root / REFERENCE_SETUP_FILE
        self.log_path = self.root / REFERENCE_SETUP_LOG_FILE

    def save_camera_calibration(
        self,
        result: CameraCalibrationSolveResult,
        *,
        camera_description: str,
        camera_fps: float,
        tag_size_m: float,
        thigh_tag_id: int,
        shank_tag_id: int,
        specification: CheckerboardSpecification,
        frame_rotation: str = "none",
    ) -> dict[str, Any]:
        calibration = result.calibration
        saved_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        payload: dict[str, Any] = {
            "schema_version": REFERENCE_SETUP_SCHEMA_VERSION,
            "setup_version": "reference_setup_v2_2",
            "saved_utc": saved_utc,
            "camera": {
                "description": str(camera_description),
                "resolution": {
                    "width": calibration.image_width,
                    "height": calibration.image_height,
                },
                "fps": float(camera_fps),
                "frame_rotation": str(frame_rotation),
                "camera_matrix": [list(row) for row in calibration.camera_matrix],
                "distortion_coefficients": list(calibration.distortion_coefficients),
            },
            "checkerboard": {
                "inner_corners_x": int(specification.inner_corners_x),
                "inner_corners_y": int(specification.inner_corners_y),
                "square_size_mm": float(specification.square_size_m * 1000.0),
                "accepted_views": int(result.accepted_views),
            },
            "camera_calibration_quality": {
                "passed": True,
                "rms_reprojection_error_px": float(result.rms_reprojection_error_px),
                "mean_view_error_px": float(result.mean_view_error_px),
                "max_view_error_px": float(result.max_view_error_px),
                "max_allowed_rms_px": float(MAX_CALIBRATION_RMS_PX),
            },
            "reference": {
                "tag_family": APRILTAG_DICTIONARY_NAME,
                "tag_size_mm": float(tag_size_m * 1000.0),
                "thigh_tag_id": int(thigh_tag_id),
                "shank_tag_id": int(shank_tag_id),
            },
            "validation": {
                "camera_calibration_quality_passed": True,
                "marker_reference_known_angle_validation": "not_performed",
            },
        }
        self._atomic_write_yaml(payload)
        self._append_log(
            f"{saved_utc} | CAMERA_CALIBRATION_SAVED | "
            f"{calibration.image_width}x{calibration.image_height} | "
            f"RMS={result.rms_reprojection_error_px:.4f}px | "
            f"views={result.accepted_views} | camera={camera_description}"
        )
        return payload

    def load_camera_calibration(
        self,
        *,
        expected_width: int,
        expected_height: int,
        expected_frame_rotation: str = "none",
    ) -> ReferenceSetupLoadResult:
        if not self.yaml_path.is_file():
            return ReferenceSetupLoadResult(
                calibration=None,
                state="NOT_FOUND",
                message="No persistent camera calibration has been saved yet.",
            )
        try:
            raw = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("reference_setup.yaml root must be a mapping")
            if int(raw.get("schema_version", -1)) != REFERENCE_SETUP_SCHEMA_VERSION:
                raise ValueError("unsupported reference setup schema version")
            camera = raw.get("camera")
            quality = raw.get("camera_calibration_quality")
            if not isinstance(camera, Mapping) or not isinstance(quality, Mapping):
                raise ValueError("camera or camera_calibration_quality section is missing")
            if quality.get("passed") is not True:
                raise ValueError("stored camera calibration is not marked as quality-passed")
            rms = float(quality.get("rms_reprojection_error_px"))
            if not (0.0 <= rms <= MAX_CALIBRATION_RMS_PX):
                raise ValueError("stored reprojection RMS fails the fixed quality gate")
            resolution = camera.get("resolution")
            if not isinstance(resolution, Mapping):
                raise ValueError("camera resolution is missing")
            width = int(resolution.get("width", 0))
            height = int(resolution.get("height", 0))
            calibration = CameraCalibration.from_values(
                camera.get("camera_matrix"),
                camera.get("distortion_coefficients"),
                image_width=width,
                image_height=height,
            )
            saved_utc = str(raw.get("saved_utc", "") or "")
            camera_description = str(camera.get("description", "") or "")
            stored_rotation = str(camera.get("frame_rotation", "none") or "none")
            if stored_rotation != str(expected_frame_rotation):
                return ReferenceSetupLoadResult(
                    calibration=None,
                    state="FRAME_ROTATION_MISMATCH",
                    message=(
                        f"Saved camera calibration uses frame rotation {stored_rotation!r}, but the active "
                        f"rotation is {str(expected_frame_rotation)!r}. Restore the matching rotation or recalibrate."
                    ),
                    saved_utc=saved_utc,
                    camera_description=camera_description,
                    rms_reprojection_error_px=rms,
                )
            if width != int(expected_width) or height != int(expected_height):
                return ReferenceSetupLoadResult(
                    calibration=None,
                    state="RESOLUTION_MISMATCH",
                    message=(
                        f"Saved camera calibration is {width}x{height}, but the active camera "
                        f"is {int(expected_width)}x{int(expected_height)}. Recalibrate or restore the matching profile."
                    ),
                    saved_utc=saved_utc,
                    camera_description=camera_description,
                    rms_reprojection_error_px=rms,
                )
            return ReferenceSetupLoadResult(
                calibration=calibration,
                state="LOADED",
                message="Persistent camera calibration loaded and resolution-validated.",
                saved_utc=saved_utc,
                camera_description=camera_description,
                rms_reprojection_error_px=rms,
            )
        except Exception as exc:
            return ReferenceSetupLoadResult(
                calibration=None,
                state="INVALID",
                message=f"Persistent camera calibration is invalid: {exc}",
            )

    def _atomic_write_yaml(self, payload: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.yaml_path.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.yaml_path)

    def _append_log(self, line: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(str(line).rstrip() + "\n")
