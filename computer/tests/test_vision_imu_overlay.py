from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from computer.vision_imu_calibration.geometry import KneeAngleResult
from computer.vision_imu_calibration.models import Landmark, LegLandmarks, LegSide
from computer.vision_imu_calibration.overlay import draw_leg_overlay


def _landmark(x: float, y: float) -> Landmark:
    return Landmark(x=x, y=y, visibility=1.0, presence=1.0)


def _leg() -> LegLandmarks:
    return LegLandmarks(
        side=LegSide.RIGHT,
        hip=_landmark(0.62, 0.25),
        knee=_landmark(0.52, 0.54),
        ankle=_landmark(0.72, 0.80),
    )


def test_valid_overlay_restores_pose_bars_joints_arc_and_status_badge() -> None:
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    angle = KneeAngleResult(angle_deg=120.0, valid=True, confidence=0.9, reason="ok")

    output = draw_leg_overlay(
        frame,
        _leg(),
        angle,
        inference_fps=15.0,
        source_description="camera:0",
        recording_active=True,
    )

    # Compact tracking badge remains in the upper-left.
    assert np.any(output[:80, :260] != 0)
    # Pose bars/joints/arc modify the central/lower body region as well.
    assert np.any(output[90:, 180:] != 0)
    assert not np.array_equal(output, frame)


def test_invalid_tracking_keeps_badge_but_hides_untrusted_pose_geometry() -> None:
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    invalid = KneeAngleResult(angle_deg=None, valid=False, confidence=0.1, reason="low_confidence")

    output = draw_leg_overlay(
        frame,
        _leg(),
        invalid,
        inference_fps=0.0,
        source_description="",
    )

    assert np.any(output[:80, :260] != 0)
    assert np.all(output[100:, :] == 0)


def test_valid_and_invalid_tracking_badges_are_visually_distinct() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    valid = KneeAngleResult(angle_deg=170.0, valid=True, confidence=0.9, reason="ok")
    invalid = KneeAngleResult(angle_deg=None, valid=False, confidence=0.1, reason="low_confidence")

    valid_output = draw_leg_overlay(frame, None, valid, inference_fps=0.0, source_description="")
    invalid_output = draw_leg_overlay(frame, None, invalid, inference_fps=0.0, source_description="")

    assert not np.array_equal(valid_output, invalid_output)
