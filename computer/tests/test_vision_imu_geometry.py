from __future__ import annotations

import pytest

from computer.vision_imu_calibration.geometry import calculate_knee_flexion_deg
from computer.vision_imu_calibration.models import Landmark, LegLandmarks, LegSide, PoseFrame


def landmark(x: float, y: float, confidence: float = 1.0) -> Landmark:
    return Landmark(x=x, y=y, visibility=confidence, presence=confidence)


def leg(
    hip: tuple[float, float],
    knee: tuple[float, float],
    ankle: tuple[float, float],
    confidence: float = 1.0,
) -> LegLandmarks:
    return LegLandmarks(
        side=LegSide.LEFT,
        hip=landmark(*hip, confidence),
        knee=landmark(*knee, confidence),
        ankle=landmark(*ankle, confidence),
    )


def test_straight_leg_is_one_hundred_eighty_degrees() -> None:
    result = calculate_knee_flexion_deg(leg((0.5, 0.2), (0.5, 0.5), (0.5, 0.8)))
    assert result.valid
    assert result.angle_deg == pytest.approx(180.0, abs=1e-9)


def test_right_angle_is_ninety_degrees() -> None:
    result = calculate_knee_flexion_deg(leg((0.5, 0.2), (0.5, 0.5), (0.8, 0.5)))
    assert result.valid
    assert result.angle_deg == pytest.approx(90.0, abs=1e-9)


def test_mirroring_does_not_change_angle() -> None:
    left = calculate_knee_flexion_deg(leg((0.4, 0.2), (0.4, 0.5), (0.7, 0.65)))
    right = calculate_knee_flexion_deg(leg((0.6, 0.2), (0.6, 0.5), (0.3, 0.65)))
    assert left.angle_deg == pytest.approx(right.angle_deg, abs=1e-9)


def test_low_confidence_is_rejected() -> None:
    result = calculate_knee_flexion_deg(
        leg((0.5, 0.2), (0.5, 0.5), (0.5, 0.8), confidence=0.4)
    )
    assert not result.valid
    assert result.angle_deg is None
    assert result.reason == "low_confidence"


def test_degenerate_segment_is_rejected() -> None:
    result = calculate_knee_flexion_deg(leg((0.5, 0.5), (0.5, 0.5), (0.5, 0.8)))
    assert not result.valid
    assert result.reason == "degenerate_segment"


def test_missing_pose_is_rejected() -> None:
    result = calculate_knee_flexion_deg(None)
    assert not result.valid
    assert result.reason == "pose_missing"


def test_auto_leg_selects_higher_confidence_chain_after_side_swap() -> None:
    pose = PoseFrame(
        timestamp_ms=1,
        landmarks={
            "left_hip": landmark(0.2, 0.2, 0.7),
            "left_knee": landmark(0.2, 0.5, 0.7),
            "left_ankle": landmark(0.2, 0.8, 0.7),
            "right_hip": landmark(0.8, 0.2, 0.95),
            "right_knee": landmark(0.8, 0.5, 0.95),
            "right_ankle": landmark(0.8, 0.8, 0.95),
        },
    )
    selected = pose.leg(LegSide.AUTO)
    assert selected is not None
    assert selected.side is LegSide.LEFT


def test_explicit_left_side_uses_swapped_landmarks() -> None:
    pose = PoseFrame(
        timestamp_ms=1,
        landmarks={
            "left_hip": landmark(0.2, 0.2, 0.7),
            "left_knee": landmark(0.2, 0.5, 0.7),
            "left_ankle": landmark(0.2, 0.8, 0.7),
            "right_hip": landmark(0.8, 0.2, 0.95),
            "right_knee": landmark(0.8, 0.5, 0.95),
            "right_ankle": landmark(0.8, 0.8, 0.95),
        },
    )
    selected = pose.leg(LegSide.LEFT)
    assert selected is not None
    assert selected.knee.x == pytest.approx(0.8)


def test_angle_is_clamped_to_physical_range() -> None:
    result = calculate_knee_flexion_deg(leg((0.5, 0.2), (0.5, 0.5), (0.5, 0.2)))
    assert result.valid
    assert result.angle_deg == pytest.approx(0.0)
    assert 0.0 <= result.angle_deg <= 180.0
