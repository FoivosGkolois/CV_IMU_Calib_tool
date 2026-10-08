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


@pytest.mark.parametrize("width,height", [(640, 480), (1280, 720), (480, 640)])
def test_oblique_right_angle_is_resolution_independent(width: int, height: int) -> None:
    # Both segments are diagonal and perpendicular in pixel coordinates.
    knee_x, knee_y = 0.5, 0.5
    offset_px = 90.0
    subject = leg(
        (knee_x + offset_px / width, knee_y - offset_px / height),
        (knee_x, knee_y),
        (knee_x + offset_px / width, knee_y + offset_px / height),
    )
    corrected = calculate_knee_flexion_deg(subject, image_aspect_ratio=width / height)
    assert corrected.valid
    assert corrected.angle_deg == pytest.approx(90.0, abs=1e-9)


def test_oblique_angle_requires_real_image_aspect_ratio() -> None:
    subject = leg((0.6, 0.3), (0.5, 0.5), (0.6, 0.7))
    corrected = calculate_knee_flexion_deg(subject, image_aspect_ratio=2.0)
    legacy = calculate_knee_flexion_deg(subject)
    assert corrected.angle_deg == pytest.approx(90.0, abs=1e-9)
    assert legacy.angle_deg != pytest.approx(90.0, abs=1.0)


@pytest.mark.parametrize("bad_ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_image_aspect_ratio_is_rejected(bad_ratio: float) -> None:
    with pytest.raises(ValueError, match="image_aspect_ratio"):
        calculate_knee_flexion_deg(
            leg((0.5, 0.2), (0.5, 0.5), (0.5, 0.8)),
            image_aspect_ratio=bad_ratio,
        )


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
