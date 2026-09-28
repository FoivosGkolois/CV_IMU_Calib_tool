from __future__ import annotations

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("OpenCV aruco module is unavailable", allow_module_level=True)

from computer.vision_imu_calibration.marker_reference import (
    APRILTAG_DICTIONARY_NAME,
    SHANK_TAG_ID,
    TAG_SIZE_M,
    THIGH_TAG_ID,
    AprilTagReference,
    CameraCalibration,
    MarkerDetection,
)


def _canvas_with_markers(ids: list[int]) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, APRILTAG_DICTIONARY_NAME)
    )
    canvas = np.full((600, 1000, 3), 255, dtype=np.uint8)
    for index, tag_id in enumerate(ids):
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 240)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        x = 120 + index * 500
        y = 170
        canvas[y : y + 240, x : x + 240] = marker_bgr
    return canvas


def _camera_calibration(width: int = 1280, height: int = 720) -> CameraCalibration:
    return CameraCalibration.from_values(
        [[900.0, 0.0, width / 2.0], [0.0, 900.0, height / 2.0], [0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        image_width=width,
        image_height=height,
    )


def _rotation_vector(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    axis_np = np.asarray(axis, dtype=float)
    axis_np /= np.linalg.norm(axis_np)
    return (axis_np * math.radians(angle_deg)).reshape(3, 1)


def _rotation_matrix(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(_rotation_vector(axis, angle_deg))
    return matrix


def _project_detection(
    tag_id: int,
    rotation_matrix: np.ndarray,
    translation_m: tuple[float, float, float],
    calibration: CameraCalibration,
) -> MarkerDetection:
    half = TAG_SIZE_M / 2.0
    object_points = np.asarray(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        np.asarray(translation_m, dtype=float).reshape(3, 1),
        np.asarray(calibration.camera_matrix, dtype=float),
        np.asarray(calibration.distortion_coefficients, dtype=float),
    )
    return MarkerDetection(
        tag_id=tag_id,
        corners=tuple((float(x), float(y)) for x, y in image_points.reshape(-1, 2)),
    )


def test_apriltag_reference_detects_fixed_thigh_and_shank_ids() -> None:
    reference = AprilTagReference()
    reference.open()
    result = reference.process(_canvas_with_markers([THIGH_TAG_ID, SHANK_TAG_ID]))

    assert result.thigh_detected
    assert result.shank_detected
    assert result.quality == pytest.approx(1.0)
    assert result.valid is False
    assert result.angle_deg is None
    assert result.reason == "reference_setup_required"
    reference.close()


def test_apriltag_reference_reports_missing_required_tag() -> None:
    reference = AprilTagReference()
    reference.open()
    result = reference.process(_canvas_with_markers([THIGH_TAG_ID]))

    assert result.thigh_detected
    assert not result.shank_detected
    assert result.reason == "missing_shank_tag"
    assert result.valid is False
    reference.close()


def test_calibrated_3d_geometry_returns_expected_posterior_knee_angle() -> None:
    calibration = _camera_calibration()
    reference = AprilTagReference(calibration)
    reference.open()

    common_body_rotation = _rotation_matrix((0.0, 1.0, 0.0), 25.0)
    thigh_rotation = common_body_rotation
    shank_rotation = common_body_rotation @ _rotation_matrix((0.0, 0.0, 1.0), 30.0)
    thigh = _project_detection(
        THIGH_TAG_ID,
        thigh_rotation,
        (-0.10, 0.12, 1.20),
        calibration,
    )
    shank = _project_detection(
        SHANK_TAG_ID,
        shank_rotation,
        (0.10, -0.12, 1.20),
        calibration,
    )

    result = reference.evaluate_detections(
        [thigh, shank],
        frame_shape=(calibration.image_height, calibration.image_width),
    )

    assert result.valid is True
    assert result.reason == "ok_3d_reference"
    assert result.angle_deg == pytest.approx(150.0, abs=0.15)
    assert result.thigh_pose is not None and result.thigh_pose.valid
    assert result.shank_pose is not None and result.shank_pose.valid
    reference.close()


def test_calibrated_reference_rejects_resolution_mismatch() -> None:
    calibration = _camera_calibration()
    reference = AprilTagReference(calibration)
    reference.open()
    identity = np.eye(3)
    thigh = _project_detection(THIGH_TAG_ID, identity, (-0.1, 0.1, 1.0), calibration)
    shank = _project_detection(SHANK_TAG_ID, identity, (0.1, -0.1, 1.0), calibration)

    result = reference.evaluate_detections([thigh, shank], frame_shape=(480, 640))

    assert result.valid is False
    assert result.angle_deg is None
    assert result.reason == "camera_resolution_mismatch"
    reference.close()


def test_calibrated_reference_rejects_too_oblique_pose() -> None:
    calibration = _camera_calibration()
    reference = AprilTagReference(calibration)
    reference.open()
    thigh = _project_detection(
        THIGH_TAG_ID,
        _rotation_matrix((0.0, 1.0, 0.0), 72.0),
        (-0.1, 0.1, 0.8),
        calibration,
    )
    shank = _project_detection(
        SHANK_TAG_ID,
        _rotation_matrix((0.0, 1.0, 0.0), 20.0),
        (0.1, -0.1, 0.8),
        calibration,
    )

    result = reference.evaluate_detections(
        [thigh, shank],
        frame_shape=(calibration.image_height, calibration.image_width),
    )

    assert result.valid is False
    assert result.reason == "thigh_pose_view_too_oblique"
    reference.close()


def test_generated_tag_image_is_deterministic_binary_marker() -> None:
    reference = AprilTagReference()
    image = reference.generate_marker_image(THIGH_TAG_ID, 256)
    assert image.shape == (256, 256)
    assert set(np.unique(image)).issubset({0, 255})
    reference.close()
