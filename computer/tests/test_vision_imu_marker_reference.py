from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("OpenCV aruco module is unavailable", allow_module_level=True)

from computer.vision_imu_calibration.marker_reference import (
    APRILTAG_DICTIONARY_NAME,
    SHANK_TAG_ID,
    THIGH_TAG_ID,
    AprilTagReference,
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


def test_generated_tag_image_is_deterministic_binary_marker() -> None:
    reference = AprilTagReference()
    image = reference.generate_marker_image(THIGH_TAG_ID, 256)
    assert image.shape == (256, 256)
    assert set(np.unique(image)).issubset({0, 255})
    reference.close()
