from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from computer.vision_imu_calibration.marker_reference import MarkerDetection, MarkerReferenceResult
from computer.vision_imu_calibration.overlay import draw_marker_overlay


def _detection(tag_id: int, x: float) -> MarkerDetection:
    return MarkerDetection(
        tag_id=tag_id,
        corners=((x, 100.0), (x + 80.0, 100.0), (x + 80.0, 180.0), (x, 180.0)),
    )


def test_overlay_draws_detected_tags_and_status_badge() -> None:
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    result = MarkerReferenceResult(
        angle_deg=None,
        valid=False,
        reason="reference_setup_required",
        quality=1.0,
        thigh=_detection(0, 90.0),
        shank=_detection(1, 300.0),
    )

    output = draw_marker_overlay(
        frame,
        result,
        processing_fps=30.0,
        source_description="camera:0",
        recording_active=True,
    )

    assert np.any(output[:80, :360] != 0)
    assert np.any(output[80:220, :] != 0)
    assert not np.array_equal(output, frame)


def test_overlay_with_no_tags_only_draws_invalid_badge() -> None:
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    result = MarkerReferenceResult(
        angle_deg=None,
        valid=False,
        reason="missing_both_tags",
        quality=0.0,
        thigh=None,
        shank=None,
    )

    output = draw_marker_overlay(frame, result, processing_fps=0.0, source_description="")
    assert np.any(output[:80, :320] != 0)
    assert np.all(output[100:, :] == 0)
