from __future__ import annotations

import pytest

from computer.vision_imu_calibration.models import LegSide
from computer.vision_imu_calibration.tuning import FrameRotation, RuntimeTuning


def test_runtime_tuning_accepts_valid_live_values() -> None:
    tuning = RuntimeTuning.from_mapping(
        {
            "leg_side": "right",
            "frame_rotation": "clockwise_90",
            "confidence_threshold": 0.61,
            "landmark_smoothing": 0.58,
            "angle_smoothing": 0.42,
            "min_segment_length": 0.03,
            "max_angle_jump_deg": 36,
        }
    )
    assert tuning.leg_side is LegSide.RIGHT
    assert tuning.frame_rotation is FrameRotation.CLOCKWISE_90
    assert tuning.confidence_threshold == pytest.approx(0.61)
    assert tuning.landmark_smoothing == pytest.approx(0.58)
    assert tuning.angle_smoothing == pytest.approx(0.42)
    assert tuning.min_segment_length == pytest.approx(0.03)
    assert tuning.max_angle_jump_deg == pytest.approx(36)


def test_runtime_tuning_partial_update_preserves_other_values() -> None:
    current = RuntimeTuning(confidence_threshold=0.7, angle_smoothing=0.2)
    updated = RuntimeTuning.from_mapping({"angle_smoothing": 0.8}, current=current)
    assert updated.confidence_threshold == pytest.approx(0.7)
    assert updated.angle_smoothing == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_threshold", 1.5),
        ("landmark_smoothing", 1.5),
        ("angle_smoothing", -0.1),
        ("min_segment_length", 0.0),
        ("max_angle_jump_deg", 181),
        ("leg_side", "both"),
        ("frame_rotation", "sideways"),
    ],
)
def test_runtime_tuning_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        RuntimeTuning.from_mapping({field: value})


def test_default_frame_rotation_matches_portrait_mounting() -> None:
    from computer.vision_imu_calibration.tuning import FrameRotation, RuntimeTuning

    assert RuntimeTuning().frame_rotation is FrameRotation.COUNTERCLOCKWISE_90
