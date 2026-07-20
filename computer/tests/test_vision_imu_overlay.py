from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from computer.vision_imu_calibration.geometry import KneeAngleResult
from computer.vision_imu_calibration.overlay import draw_leg_overlay


def test_overlay_changes_only_compact_top_left_status_area() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    angle = KneeAngleResult(angle_deg=180.0, valid=True, confidence=0.9, reason="ok")

    output = draw_leg_overlay(
        frame,
        None,
        angle,
        inference_fps=15.0,
        source_description="camera:0",
        recording_active=True,
    )

    assert np.any(output[:80, :260] != 0)
    assert np.all(output[100:, :] == 0)
    assert np.all(output[:, 280:] == 0)


def test_invalid_tracking_uses_a_distinct_status_badge() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    valid = KneeAngleResult(angle_deg=170.0, valid=True, confidence=0.9, reason="ok")
    invalid = KneeAngleResult(angle_deg=None, valid=False, confidence=0.1, reason="low_confidence")

    valid_output = draw_leg_overlay(
        frame,
        None,
        valid,
        inference_fps=0.0,
        source_description="",
    )
    invalid_output = draw_leg_overlay(
        frame,
        None,
        invalid,
        inference_fps=0.0,
        source_description="",
    )

    assert not np.array_equal(valid_output, invalid_output)
