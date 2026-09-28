from __future__ import annotations

import pytest

from computer.vision_imu_calibration.tuning import FrameRotation, RuntimeTuning


def test_runtime_tuning_only_exposes_frame_rotation() -> None:
    tuning = RuntimeTuning.from_mapping({"frame_rotation": "clockwise_90"})
    assert tuning.frame_rotation is FrameRotation.CLOCKWISE_90
    assert tuning.to_dict() == {"frame_rotation": "clockwise_90"}


def test_runtime_tuning_partial_update_preserves_rotation() -> None:
    current = RuntimeTuning(frame_rotation=FrameRotation.ROTATE_180)
    updated = RuntimeTuning.from_mapping({}, current=current)
    assert updated.frame_rotation is FrameRotation.ROTATE_180


def test_runtime_tuning_rejects_invalid_rotation() -> None:
    with pytest.raises(ValueError):
        RuntimeTuning.from_mapping({"frame_rotation": "sideways"})


def test_default_frame_rotation_matches_portrait_mounting() -> None:
    assert RuntimeTuning().frame_rotation is FrameRotation.COUNTERCLOCKWISE_90
