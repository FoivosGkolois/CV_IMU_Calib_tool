from __future__ import annotations

import pytest

from computer.vision_imu_calibration.landmark_filter import LegLandmarkEmaFilter
from computer.vision_imu_calibration.models import Landmark, LegLandmarks, LegSide


def _landmark(x: float, y: float, confidence: float = 0.9) -> Landmark:
    return Landmark(x=x, y=y, visibility=confidence, presence=confidence)


def _leg(side: LegSide, offset: float, confidence: float = 0.9) -> LegLandmarks:
    return LegLandmarks(
        side=side,
        hip=_landmark(offset + 0.0, 0.1, confidence),
        knee=_landmark(offset + 0.1, 0.5, confidence),
        ankle=_landmark(offset + 0.2, 0.9, confidence),
    )


def test_landmark_filter_smooths_coordinates_and_keeps_current_confidence() -> None:
    filter_ = LegLandmarkEmaFilter()
    first = filter_.apply(_leg(LegSide.RIGHT, 0.0), smoothing=0.75)
    second = filter_.apply(_leg(LegSide.RIGHT, 0.4, confidence=0.6), smoothing=0.75)

    assert first is not None and second is not None
    assert second.hip.x == pytest.approx(0.1)
    assert second.knee.x == pytest.approx(0.2)
    assert second.hip.confidence == pytest.approx(0.6)


def test_landmark_filter_resets_when_tracked_side_changes() -> None:
    filter_ = LegLandmarkEmaFilter()
    filter_.apply(_leg(LegSide.LEFT, 0.0), smoothing=0.9)
    right = filter_.apply(_leg(LegSide.RIGHT, 0.5), smoothing=0.9)

    assert right is not None
    assert right.hip.x == pytest.approx(0.5)


def test_landmark_filter_resets_on_missing_tracking() -> None:
    filter_ = LegLandmarkEmaFilter()
    filter_.apply(_leg(LegSide.RIGHT, 0.0), smoothing=0.9)
    assert filter_.apply(None, smoothing=0.9) is None
    recovered = filter_.apply(_leg(LegSide.RIGHT, 0.7), smoothing=0.9)
    assert recovered is not None
    assert recovered.hip.x == pytest.approx(0.7)
