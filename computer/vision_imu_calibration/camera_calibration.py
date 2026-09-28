from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .marker_reference import CameraCalibration


CHECKERBOARD_INNER_CORNERS_X = 9
CHECKERBOARD_INNER_CORNERS_Y = 6
CHECKERBOARD_SQUARE_SIZE_M = 0.025
REQUIRED_CALIBRATION_VIEWS = 15
MIN_BOARD_AREA_RATIO = 0.035
MAX_BOARD_AREA_RATIO = 0.80
MIN_VIEW_FEATURE_DISTANCE = 0.085
MIN_CAPTURE_INTERVAL_S = 0.30
MAX_CALIBRATION_RMS_PX = 1.50


@dataclass(frozen=True)
class CheckerboardSpecification:
    inner_corners_x: int = CHECKERBOARD_INNER_CORNERS_X
    inner_corners_y: int = CHECKERBOARD_INNER_CORNERS_Y
    square_size_m: float = CHECKERBOARD_SQUARE_SIZE_M

    @property
    def pattern_size(self) -> tuple[int, int]:
        return int(self.inner_corners_x), int(self.inner_corners_y)

    @property
    def corner_count(self) -> int:
        return int(self.inner_corners_x * self.inner_corners_y)


@dataclass(frozen=True)
class CameraCalibrationSolveResult:
    calibration: CameraCalibration
    rms_reprojection_error_px: float
    mean_view_error_px: float
    max_view_error_px: float
    accepted_views: int


class CheckerboardCalibrationSession:
    """One checkerboard calibration capture/solve session.

    Persistence is owned by the runtime/reference-setup store so this class
    remains focused on frame acceptance and camera calibration mathematics.
    """

    def __init__(
        self,
        *,
        specification: CheckerboardSpecification | None = None,
        required_views: int = REQUIRED_CALIBRATION_VIEWS,
    ) -> None:
        self.specification = specification or CheckerboardSpecification()
        self.required_views = max(8, int(required_views))
        self._lock = threading.RLock()
        self._active = False
        self._state = "IDLE"
        self._message = "Start checkerboard calibration when the printed board is ready."
        self._accepted_points: list[Any] = []
        self._accepted_features: list[tuple[float, float, float, float, float]] = []
        self._image_size: tuple[int, int] | None = None
        self._last_detection = False
        self._last_rejection_reason = ""
        self._last_accept_monotonic = 0.0
        self._latest_corners: Any | None = None
        self._result: CameraCalibrationSolveResult | None = None

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._active:
                raise RuntimeError("Checkerboard camera calibration is already running")
            self._active = True
            self._state = "CAPTURING"
            self._message = (
                "Move the checkerboard through different positions and tilts. "
                "Good views are captured automatically."
            )
            self._accepted_points = []
            self._accepted_features = []
            self._image_size = None
            self._last_detection = False
            self._last_rejection_reason = "waiting_for_checkerboard"
            self._last_accept_monotonic = 0.0
            self._latest_corners = None
            self._result = None
            return self._status_unlocked()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            self._active = False
            self._state = "IDLE"
            self._message = "Checkerboard calibration cancelled."
            self._accepted_points = []
            self._accepted_features = []
            self._image_size = None
            self._last_detection = False
            self._last_rejection_reason = "cancelled"
            self._last_accept_monotonic = 0.0
            self._latest_corners = None
            self._result = None
            return self._status_unlocked()

    def invalidate_for_camera_change(self) -> None:
        with self._lock:
            self._active = False
            self._state = "IDLE"
            self._message = "Camera geometry changed; run checkerboard calibration again."
            self._accepted_points = []
            self._accepted_features = []
            self._image_size = None
            self._last_detection = False
            self._last_rejection_reason = "camera_geometry_changed"
            self._last_accept_monotonic = 0.0
            self._latest_corners = None
            self._result = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict[str, Any]:
        result = self._result
        width, height = self._image_size or (None, None)
        return {
            "active": self._active,
            "state": self._state,
            "message": self._message,
            "accepted_views": len(self._accepted_points),
            "required_views": self.required_views,
            "checkerboard_detected": self._last_detection,
            "last_rejection_reason": self._last_rejection_reason,
            "image_width": width,
            "image_height": height,
            "rms_reprojection_error_px": (
                result.rms_reprojection_error_px if result is not None else None
            ),
            "mean_view_error_px": result.mean_view_error_px if result is not None else None,
            "max_view_error_px": result.max_view_error_px if result is not None else None,
            "applied_to_reference": result is not None,
            "persistent": False,
            "specification": {
                "inner_corners_x": self.specification.inner_corners_x,
                "inner_corners_y": self.specification.inner_corners_y,
                "square_size_mm": self.specification.square_size_m * 1000.0,
            },
            "quality_gates": {
                "min_board_area_ratio": MIN_BOARD_AREA_RATIO,
                "max_board_area_ratio": MAX_BOARD_AREA_RATIO,
                "min_view_feature_distance": MIN_VIEW_FEATURE_DISTANCE,
                "max_rms_reprojection_error_px": MAX_CALIBRATION_RMS_PX,
            },
        }

    def process_frame(self, frame_bgr: Any) -> CameraCalibration | None:
        with self._lock:
            if not self._active:
                return None

        cv2, np = _load_cv2_numpy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        height, width = int(gray.shape[0]), int(gray.shape[1])
        image_size = (width, height)

        flags = int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
        flags |= int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
        found, corners = cv2.findChessboardCornersSB(
            gray,
            self.specification.pattern_size,
            flags=flags,
        )

        now = time.monotonic()
        with self._lock:
            if not self._active:
                return None
            if self._image_size is None:
                self._image_size = image_size
            elif self._image_size != image_size:
                self._fail_unlocked("Camera resolution changed during checkerboard calibration.")
                return None

            self._last_detection = bool(found)
            self._latest_corners = corners.copy() if found and corners is not None else None
            if not found or corners is None:
                self._last_rejection_reason = "checkerboard_not_detected"
                return None

            points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
            if points.shape != (self.specification.corner_count, 2):
                self._last_rejection_reason = "unexpected_corner_count"
                return None

            feature, reason = _view_feature(points, image_size, cv2)
            if reason:
                self._last_rejection_reason = reason
                return None
            if now - self._last_accept_monotonic < MIN_CAPTURE_INTERVAL_S:
                self._last_rejection_reason = "hold_or_move_board"
                return None
            if any(
                _feature_distance(feature, previous) < MIN_VIEW_FEATURE_DISTANCE
                for previous in self._accepted_features
            ):
                self._last_rejection_reason = "view_too_similar"
                return None

            self._accepted_points.append(points.reshape(-1, 1, 2).copy())
            self._accepted_features.append(feature)
            self._last_accept_monotonic = now
            self._last_rejection_reason = "accepted"
            accepted = len(self._accepted_points)
            self._message = (
                f"Accepted view {accepted}/{self.required_views}. "
                "Move or tilt the board for the next view."
            )
            if accepted < self.required_views:
                return None

            try:
                result = calibrate_checkerboard_views(
                    self._accepted_points,
                    image_size=image_size,
                    specification=self.specification,
                )
            except Exception as exc:
                self._fail_unlocked(f"Camera calibration failed: {exc}")
                return None

            self._result = result
            self._active = False
            self._state = "COMPLETE"
            self._message = (
                "Camera calibration complete and quality-passed."
            )
            return result.calibration

    def result(self) -> CameraCalibrationSolveResult | None:
        with self._lock:
            return self._result

    def draw_overlay(self, frame_bgr: Any) -> Any:
        cv2, _np = _load_cv2_numpy()
        with self._lock:
            active = self._active
            corners = None if self._latest_corners is None else self._latest_corners.copy()
            accepted = len(self._accepted_points)
            required = self.required_views
            detected = self._last_detection
        if not active:
            return frame_bgr
        if corners is not None:
            cv2.drawChessboardCorners(
                frame_bgr,
                self.specification.pattern_size,
                corners,
                detected,
            )
        label = f"CAMERA CAL {accepted}/{required}"
        cv2.putText(
            frame_bgr,
            label,
            (18, max(28, int(frame_bgr.shape[0]) - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    def _fail_unlocked(self, message: str) -> None:
        self._active = False
        self._state = "FAILED"
        self._message = str(message)
        self._last_rejection_reason = "calibration_failed"


def calibrate_checkerboard_views(
    image_points: Sequence[Any],
    *,
    image_size: tuple[int, int],
    specification: CheckerboardSpecification | None = None,
) -> CameraCalibrationSolveResult:
    cv2, np = _load_cv2_numpy()
    spec = specification or CheckerboardSpecification()
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if len(image_points) < 8:
        raise ValueError("at least 8 checkerboard views are required")

    object_template = np.zeros((spec.corner_count, 3), np.float32)
    grid = np.mgrid[0 : spec.inner_corners_x, 0 : spec.inner_corners_y].T.reshape(-1, 2)
    object_template[:, :2] = grid.astype(np.float32) * float(spec.square_size_m)

    normalized_image_points: list[Any] = []
    object_points: list[Any] = []
    for view in image_points:
        points = np.asarray(view, dtype=np.float32).reshape(-1, 1, 2)
        if points.shape[0] != spec.corner_count:
            raise ValueError("checkerboard view has the wrong number of corners")
        if not np.all(np.isfinite(points)):
            raise ValueError("checkerboard view contains non-finite coordinates")
        normalized_image_points.append(points)
        object_points.append(object_template.copy())

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        normalized_image_points,
        (width, height),
        None,
        None,
    )
    rms = float(rms)
    if not math.isfinite(rms):
        raise RuntimeError("calibration returned a non-finite reprojection error")
    if rms > MAX_CALIBRATION_RMS_PX:
        raise RuntimeError(
            f"reprojection RMS {rms:.2f} px exceeds the fixed {MAX_CALIBRATION_RMS_PX:.2f} px gate"
        )

    matrix = np.asarray(camera_matrix, dtype=float)
    distortion_array = np.asarray(distortion, dtype=float).reshape(-1)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if not (0.30 * width <= fx <= 10.0 * width and 0.30 * height <= fy <= 10.0 * height):
        raise RuntimeError("calibration produced implausible focal lengths")
    if not (-0.10 * width <= cx <= 1.10 * width and -0.10 * height <= cy <= 1.10 * height):
        raise RuntimeError("calibration produced an implausible principal point")

    per_view_errors: list[float] = []
    for object_view, image_view, rvec, tvec in zip(
        object_points, normalized_image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(object_view, rvec, tvec, camera_matrix, distortion)
        delta = projected.reshape(-1, 2) - image_view.reshape(-1, 2)
        per_view_errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))

    calibration = CameraCalibration.from_values(
        matrix,
        distortion_array,
        image_width=width,
        image_height=height,
    )
    return CameraCalibrationSolveResult(
        calibration=calibration,
        rms_reprojection_error_px=rms,
        mean_view_error_px=float(np.mean(per_view_errors)),
        max_view_error_px=float(np.max(per_view_errors)),
        accepted_views=len(normalized_image_points),
    )


def _view_feature(points: Any, image_size: tuple[int, int], cv2: Any) -> tuple[tuple[float, float, float, float, float], str]:
    import numpy as np

    width, height = image_size
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    area = abs(float(cv2.contourArea(hull)))
    area_ratio = area / max(1.0, float(width * height))
    if area_ratio < MIN_BOARD_AREA_RATIO:
        return (0.0, 0.0, 0.0, 0.0, 0.0), "checkerboard_too_small"
    if area_ratio > MAX_BOARD_AREA_RATIO:
        return (0.0, 0.0, 0.0, 0.0, 0.0), "checkerboard_too_large"

    center = np.mean(points, axis=0)
    row_end = points[max(0, CHECKERBOARD_INNER_CORNERS_X - 1)]
    direction = row_end - points[0]
    angle = math.atan2(float(direction[1]), float(direction[0]))
    feature = (
        float(center[0]) / float(width),
        float(center[1]) / float(height),
        math.sqrt(area_ratio),
        math.sin(angle),
        math.cos(angle),
    )
    return feature, ""


def _feature_distance(
    current: tuple[float, float, float, float, float],
    previous: tuple[float, float, float, float, float],
) -> float:
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    ds = 1.5 * (current[2] - previous[2])
    da = 0.35 * math.hypot(current[3] - previous[3], current[4] - previous[4])
    return math.sqrt(dx * dx + dy * dy + ds * ds + da * da)


def _load_cv2_numpy() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for camera calibration") from exc
    return cv2, np
