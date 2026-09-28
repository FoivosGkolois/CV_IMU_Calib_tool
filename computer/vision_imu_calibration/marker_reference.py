from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import Any, Iterable, Mapping


APRILTAG_DICTIONARY_NAME = "DICT_APRILTAG_36h11"
THIGH_TAG_ID = 0
SHANK_TAG_ID = 1
TAG_SIZE_M = 0.070

# Frozen engineering quality gates for the reference layer. These are not exposed
# as routine user tuning controls. Patch 2.1 will provide the calibrated camera
# model that activates this geometry path.
MIN_MARKER_SIDE_PX = 20.0
MAX_VIEW_ANGLE_DEG = 65.0
MAX_NORMALIZED_REPROJECTION_RMSE = 0.05


@dataclass(frozen=True)
class CameraCalibration:
    """Validated camera intrinsics required by the 3-D marker reference."""

    camera_matrix: tuple[tuple[float, float, float], ...]
    distortion_coefficients: tuple[float, ...]
    image_width: int
    image_height: int

    @classmethod
    def from_values(
        cls,
        camera_matrix: Any,
        distortion_coefficients: Any,
        *,
        image_width: int,
        image_height: int,
    ) -> "CameraCalibration":
        import numpy as np

        matrix = np.asarray(camera_matrix, dtype=float)
        distortion = np.asarray(distortion_coefficients, dtype=float).reshape(-1)
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
            raise ValueError("camera calibration contains non-finite values")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        width = int(image_width)
        height = int(image_height)
        if width <= 0 or height <= 0:
            raise ValueError("camera calibration image dimensions must be positive")
        return cls(
            camera_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
            distortion_coefficients=tuple(float(value) for value in distortion),
            image_width=width,
            image_height=height,
        )


@dataclass(frozen=True)
class MarkerDetection:
    tag_id: int
    corners: tuple[tuple[float, float], ...]

    @property
    def center(self) -> tuple[float, float]:
        x = sum(point[0] for point in self.corners) / len(self.corners)
        y = sum(point[1] for point in self.corners) / len(self.corners)
        return float(x), float(y)


@dataclass(frozen=True)
class MarkerPose:
    """One calibrated 3-D tag pose plus objective frame-quality metrics."""

    valid: bool
    reason: str
    rvec: tuple[float, float, float] | None = None
    tvec_m: tuple[float, float, float] | None = None
    rotation_matrix: tuple[tuple[float, float, float], ...] | None = None
    reprojection_rmse_px: float | None = None
    normalized_reprojection_rmse: float | None = None
    mean_side_px: float | None = None
    view_angle_deg: float | None = None


@dataclass(frozen=True)
class MarkerReferenceResult:
    """One frame of the marker-based geometric reference.

    The 3-D path is only valid when calibrated camera intrinsics matching the
    active image resolution have been configured. Without that setup, tag
    detection remains available but the scientific angle stays invalid.
    """

    angle_deg: float | None
    valid: bool
    reason: str
    quality: float
    thigh: MarkerDetection | None
    shank: MarkerDetection | None
    thigh_pose: MarkerPose | None = None
    shank_pose: MarkerPose | None = None

    @property
    def thigh_detected(self) -> bool:
        return self.thigh is not None

    @property
    def shank_detected(self) -> bool:
        return self.shank is not None


class AprilTagReference:
    """Two-tag calibrated geometric reference for the knee angle.

    Measurement contract:
      - AprilTag 36h11, Tag 0 -> thigh, Tag 1 -> shank.
      - Printed black square side -> 70 mm.
      - The image-top direction of each tag is mounted along the proximal
        longitudinal direction of its body segment.
      - Knee convention -> 180 deg at collinear extension, decreasing in flexion.

    Common 3-D body rotation relative to the camera cancels because the angle is
    derived from the two segment directions in the same calibrated camera frame.
    """

    thigh_tag_id = THIGH_TAG_ID
    shank_tag_id = SHANK_TAG_ID

    def __init__(
        self,
        calibration: CameraCalibration | None = None,
        *,
        tag_size_m: float = TAG_SIZE_M,
    ) -> None:
        self._cv2: Any | None = None
        self._detector: Any | None = None
        self._dictionary: Any | None = None
        self._calibration = calibration
        self._tag_size_m = float(tag_size_m)
        if not (0.01 <= self._tag_size_m <= 0.5):
            raise ValueError("tag_size_m is outside the supported engineering range")

    @property
    def description(self) -> str:
        mode = "calibrated 3-D" if self.camera_calibrated else "detection / setup required"
        return (
            f"AprilTag 36h11 geometric reference "
            f"(thigh={self.thigh_tag_id}, shank={self.shank_tag_id}; {mode})"
        )

    @property
    def camera_calibrated(self) -> bool:
        return self._calibration is not None

    @property
    def tag_size_m(self) -> float:
        return self._tag_size_m

    def configure_camera(self, calibration: CameraCalibration) -> None:
        if not isinstance(calibration, CameraCalibration):
            raise TypeError("calibration must be a CameraCalibration")
        self._calibration = calibration

    def clear_camera_calibration(self) -> None:
        self._calibration = None

    def configuration_status(self) -> dict[str, Any]:
        calibration = self._calibration
        return {
            "camera_calibrated": calibration is not None,
            "image_width": calibration.image_width if calibration is not None else None,
            "image_height": calibration.image_height if calibration is not None else None,
            "tag_size_m": self._tag_size_m,
            "quality_gates": {
                "min_marker_side_px": MIN_MARKER_SIDE_PX,
                "max_view_angle_deg": MAX_VIEW_ANGLE_DEG,
                "max_normalized_reprojection_rmse": MAX_NORMALIZED_REPROJECTION_RMSE,
            },
        }

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed") from exc
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV AprilTag support is unavailable. Install opencv-contrib-python "
                "(or opencv-contrib-python-headless on UNO Q)."
            )
        dictionary_id = getattr(cv2.aruco, APRILTAG_DICTIONARY_NAME, None)
        if dictionary_id is None:
            raise RuntimeError(f"OpenCV does not provide {APRILTAG_DICTIONARY_NAME}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._dictionary = dictionary
        self._cv2 = cv2

    def close(self) -> None:
        self._detector = None
        self._dictionary = None
        self._cv2 = None

    def process(self, frame_bgr: Any) -> MarkerReferenceResult:
        if self._detector is None or self._cv2 is None:
            raise RuntimeError("AprilTag reference is not open")
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        detections: dict[int, MarkerDetection] = {}
        if ids is not None:
            for tag_id_raw, corner_array in zip(ids.flatten().tolist(), corners):
                tag_id = int(tag_id_raw)
                if tag_id not in {self.thigh_tag_id, self.shank_tag_id}:
                    continue
                points = corner_array.reshape(-1, 2)
                if points.shape != (4, 2):
                    continue
                detections[tag_id] = MarkerDetection(
                    tag_id=tag_id,
                    corners=tuple((float(x), float(y)) for x, y in points),
                )
        return self.evaluate_detections(detections, frame_shape=gray.shape)

    def evaluate_detections(
        self,
        detections: Mapping[int, MarkerDetection] | Iterable[MarkerDetection],
        *,
        frame_shape: tuple[int, ...],
    ) -> MarkerReferenceResult:
        """Evaluate already-detected tags; useful for deterministic replay/tests."""
        if isinstance(detections, Mapping):
            by_id = dict(detections)
        else:
            by_id = {detection.tag_id: detection for detection in detections}

        thigh = by_id.get(self.thigh_tag_id)
        shank = by_id.get(self.shank_tag_id)
        if thigh is None and shank is None:
            return self._invalid("missing_both_tags", thigh, shank, quality=0.0)
        if thigh is None:
            return self._invalid("missing_thigh_tag", thigh, shank, quality=0.5)
        if shank is None:
            return self._invalid("missing_shank_tag", thigh, shank, quality=0.5)

        calibration = self._calibration
        if calibration is None:
            return self._invalid("reference_setup_required", thigh, shank, quality=1.0)

        height, width = int(frame_shape[0]), int(frame_shape[1])
        if width != calibration.image_width or height != calibration.image_height:
            return self._invalid(
                "camera_resolution_mismatch",
                thigh,
                shank,
                quality=0.0,
            )

        thigh_pose = self._estimate_pose(thigh)
        shank_pose = self._estimate_pose(shank)
        valid_pose_count = int(thigh_pose.valid) + int(shank_pose.valid)
        quality = valid_pose_count / 2.0
        if not thigh_pose.valid:
            return self._invalid(
                f"thigh_pose_{thigh_pose.reason}",
                thigh,
                shank,
                quality=quality,
                thigh_pose=thigh_pose,
                shank_pose=shank_pose,
            )
        if not shank_pose.valid:
            return self._invalid(
                f"shank_pose_{shank_pose.reason}",
                thigh,
                shank,
                quality=quality,
                thigh_pose=thigh_pose,
                shank_pose=shank_pose,
            )

        angle_deg = self._posterior_knee_angle_deg(thigh_pose, shank_pose)
        return MarkerReferenceResult(
            angle_deg=angle_deg,
            valid=True,
            reason="ok_3d_reference",
            quality=1.0,
            thigh=thigh,
            shank=shank,
            thigh_pose=thigh_pose,
            shank_pose=shank_pose,
        )

    def _estimate_pose(self, detection: MarkerDetection) -> MarkerPose:
        import numpy as np

        if self._cv2 is None:
            raise RuntimeError("AprilTag reference is not open")
        calibration = self._calibration
        if calibration is None:
            return MarkerPose(valid=False, reason="reference_setup_required")

        image_points = np.asarray(detection.corners, dtype=np.float64)
        side_lengths = np.linalg.norm(image_points - np.roll(image_points, -1, axis=0), axis=1)
        mean_side_px = float(np.mean(side_lengths))
        if not np.isfinite(mean_side_px) or mean_side_px < MIN_MARKER_SIDE_PX:
            return MarkerPose(
                valid=False,
                reason="marker_too_small",
                mean_side_px=mean_side_px,
            )

        half = self._tag_size_m / 2.0
        object_points = np.asarray(
            [
                [-half, +half, 0.0],
                [+half, +half, 0.0],
                [+half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
        distortion = np.asarray(calibration.distortion_coefficients, dtype=np.float64)

        solved, rvecs, tvecs, _errors = self._cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=self._cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if int(solved) <= 0 or not rvecs or not tvecs:
            return MarkerPose(valid=False, reason="solvepnp_failed", mean_side_px=mean_side_px)

        candidates: list[tuple[float, Any, Any, float]] = []
        for rvec, tvec in zip(rvecs, tvecs):
            projected, _ = self._cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                camera_matrix,
                distortion,
            )
            projected = projected.reshape(-1, 2)
            rmse = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
            depth = float(np.asarray(tvec, dtype=float).reshape(3)[2])
            candidates.append((rmse, rvec, tvec, depth))

        positive_depth = [candidate for candidate in candidates if candidate[3] > 0.0]
        if not positive_depth:
            return MarkerPose(valid=False, reason="pose_behind_camera", mean_side_px=mean_side_px)
        rmse, rvec, tvec, _depth = min(positive_depth, key=lambda candidate: candidate[0])
        normalized_rmse = rmse / max(mean_side_px, 1e-9)

        rotation_matrix_np, _ = self._cv2.Rodrigues(rvec)
        tag_normal = rotation_matrix_np @ np.asarray([0.0, 0.0, 1.0], dtype=float)
        normal_z = min(1.0, max(0.0, abs(float(tag_normal[2]))))
        view_angle_deg = degrees(acos(normal_z))

        rvec_tuple = tuple(float(value) for value in np.asarray(rvec).reshape(3))
        tvec_tuple = tuple(float(value) for value in np.asarray(tvec).reshape(3))
        rotation_tuple = tuple(
            tuple(float(value) for value in row) for row in np.asarray(rotation_matrix_np)
        )
        common = {
            "rvec": rvec_tuple,
            "tvec_m": tvec_tuple,
            "rotation_matrix": rotation_tuple,
            "reprojection_rmse_px": rmse,
            "normalized_reprojection_rmse": normalized_rmse,
            "mean_side_px": mean_side_px,
            "view_angle_deg": view_angle_deg,
        }
        if normalized_rmse > MAX_NORMALIZED_REPROJECTION_RMSE:
            return MarkerPose(valid=False, reason="high_reprojection_error", **common)
        if view_angle_deg > MAX_VIEW_ANGLE_DEG:
            return MarkerPose(valid=False, reason="view_too_oblique", **common)
        return MarkerPose(valid=True, reason="ok", **common)

    @staticmethod
    def _posterior_knee_angle_deg(thigh_pose: MarkerPose, shank_pose: MarkerPose) -> float:
        import numpy as np

        if thigh_pose.rotation_matrix is None or shank_pose.rotation_matrix is None:
            raise ValueError("Both marker poses need rotation matrices")
        thigh_rotation = np.asarray(thigh_pose.rotation_matrix, dtype=float)
        shank_rotation = np.asarray(shank_pose.rotation_matrix, dtype=float)

        # +Y is the image-top direction of the printed marker and is therefore
        # mounted along each segment's proximal longitudinal direction.
        tag_proximal_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
        thigh_axis = thigh_rotation @ tag_proximal_axis
        shank_axis = shank_rotation @ tag_proximal_axis
        cosine = float(
            np.dot(thigh_axis, shank_axis)
            / (np.linalg.norm(thigh_axis) * np.linalg.norm(shank_axis))
        )
        cosine = max(-1.0, min(1.0, cosine))
        flexion_from_extension_deg = degrees(acos(cosine))
        return float(180.0 - flexion_from_extension_deg)

    @staticmethod
    def _invalid(
        reason: str,
        thigh: MarkerDetection | None,
        shank: MarkerDetection | None,
        *,
        quality: float,
        thigh_pose: MarkerPose | None = None,
        shank_pose: MarkerPose | None = None,
    ) -> MarkerReferenceResult:
        return MarkerReferenceResult(
            angle_deg=None,
            valid=False,
            reason=reason,
            quality=float(quality),
            thigh=thigh,
            shank=shank,
            thigh_pose=thigh_pose,
            shank_pose=shank_pose,
        )

    def generate_marker_image(self, tag_id: int, side_pixels: int = 1000) -> Any:
        """Generate one exact marker image for printing/tests."""
        if self._dictionary is None or self._cv2 is None:
            self.open()
        if tag_id < 0:
            raise ValueError("tag_id must be non-negative")
        side_pixels = int(side_pixels)
        if side_pixels < 64:
            raise ValueError("side_pixels must be at least 64")
        return self._cv2.aruco.generateImageMarker(self._dictionary, int(tag_id), side_pixels)
