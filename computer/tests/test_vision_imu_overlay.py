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


def test_overlay_stroke_slider_changes_bars_and_arc_but_not_joint_radius(monkeypatch) -> None:
    import cv2
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    angle = KneeAngleResult(angle_deg=120.0, valid=True, confidence=0.9, reason="ok")
    line_widths = []
    arc_widths = []
    marker_radii = []
    line_fn, poly_fn, circle_fn = cv2.line, cv2.polylines, cv2.circle

    def track_line(image, start, end, color, thickness=1, *args, **kwargs):
        line_widths.append(thickness)
        return line_fn(image, start, end, color, thickness, *args, **kwargs)

    def track_poly(image, pts, closed, color, thickness=1, *args, **kwargs):
        arc_widths.append(thickness)
        return poly_fn(image, pts, closed, color, thickness, *args, **kwargs)

    def track_circle(image, center, radius, color, thickness=1, *args, **kwargs):
        marker_radii.append(radius)
        return circle_fn(image, center, radius, color, thickness, *args, **kwargs)

    monkeypatch.setattr(cv2, "line", track_line)
    monkeypatch.setattr(cv2, "polylines", track_poly)
    monkeypatch.setattr(cv2, "circle", track_circle)
    measurements = []
    for scale in (1.0, 0.4, 1.2):
        line_widths.clear(); arc_widths.clear(); marker_radii.clear()
        draw_leg_overlay(frame, _leg(), angle, inference_fps=6.0,
                         source_description="test", stroke_scale=scale)
        measurements.append((tuple(line_widths), tuple(arc_widths), tuple(marker_radii)))
    default, slim, wide = measurements
    assert len(default[0]) == len(slim[0]) == len(wide[0]) == 2
    assert default[0][0] > slim[0][0] and wide[0][0] > default[0][0]
    assert slim[1][0] < default[1][0] < wide[1][0]
    assert len(default[2]) == 6
    assert default[2] == slim[2] == wide[2]
