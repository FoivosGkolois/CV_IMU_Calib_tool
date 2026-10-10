from __future__ import annotations

import math
from dataclasses import replace

import pytest

from computer.vision_imu_calibration.geometry import KneeAngleResult
from computer.vision_imu_calibration.models import Landmark, LegLandmarks, LegSide
from computer.vision_imu_calibration.positioning import (
    MAX_CHECK_DURATION_S,
    MIN_VALID_FRAMES,
    SAFE_AREA,
    PositioningCheck,
)


def leg() -> LegLandmarks:
    def p(x, y):
        return Landmark(x, y, 0.9, 0.9)
    return LegLandmarks(LegSide.RIGHT, p(0.35, 0.20), p(0.50, 0.50), p(0.65, 0.80))


def sweep(check, *, side_on=True, points=None, values=None):
    check.set_context(("camera", "fast", "none", "right", "right", 640, 480))
    check.begin(side_on)
    for i, value in enumerate(values or [175 - 85 * math.sin(math.pi * i / 30) for i in range(31)]):
        check.observe(i * 0.15, points or leg(), KneeAngleResult(value, True, 0.9, "ok"), 0.3)
    return check.finish(4.51)


def moving_angle(seconds: float) -> float:
    return 145.0 + 30.0 * math.cos(2.0 * math.pi * seconds / 3.0)


def run_at_fps(fps: float, *, dropout_index: int | None = None,
               spike_index: int | None = None) -> PositioningCheck:
    check = PositioningCheck()
    check.set_context(("camera", "fast", "none", "right", "right", 640, 480))
    check.begin()
    for index in range(int(MAX_CHECK_DURATION_S * fps) + 1):
        seconds = index / fps
        if index == dropout_index:
            check.observe(
                seconds, None, KneeAngleResult(None, False, 0.0, "pose_missing"), 0.3
            )
        else:
            value = 20.0 if index == spike_index else moving_angle(seconds)
            check.observe(seconds, leg(), KneeAngleResult(value, True, 0.9, "ok"), 0.3)
        if check.state != "CHECKING":
            break
    return check


def test_full_sweep_passes_without_claiming_side_on_proof():
    assert sweep(PositioningCheck())["state"] == "PASSED"
    result = sweep(PositioningCheck(), side_on=False)
    assert result["state"] == "PASSED"
    assert "operator's responsibility" in result["reason"]
    assert result["manual_side_on"] is False


@pytest.mark.parametrize("fps", [4.0, 5.0, 10.0])
def test_slow_bend_and_return_passes_at_low_and_normal_fps(fps):
    check = run_at_fps(fps)
    assert check.state == "PASSED"
    assert check.count >= MIN_VALID_FRAMES
    assert check.last_time is not None and 3.0 <= check.last_time < MAX_CHECK_DURATION_S


@pytest.mark.parametrize("mode", ["still", "bend_only"])
def test_no_motion_or_bend_only_retries_at_timeout(mode):
    check = PositioningCheck()
    check.begin()
    for index in range(41):
        seconds = index * 0.2
        value = 130.0 if mode == "still" else 175.0 - 85.0 * seconds / 8.0
        check.observe(seconds, leg(), KneeAngleResult(value, True, 0.9, "ok"), 0.3)
    assert check.state == "RETRY"
    assert ("knee motion" if mode == "still" else "bend and return") in check.reason


@pytest.mark.parametrize("field,value", [
    ("ankle", Landmark(0.99, 0.8, 1, 1)),
    ("knee", Landmark(0.5, 0.5, 0.1, 1)),
    ("hip", Landmark(0.99, 0.22, 1, 1)),
    ("ankle", Landmark(float("nan"), 0.8, 1, 1)),
    ("knee", Landmark(0.5, 0.5, float("nan"), 1)),
])
def test_persistently_bad_observations_timeout_without_counting(field, value):
    result = sweep(PositioningCheck(), points=replace(leg(), **{field: value}))
    assert result["state"] == "RETRY"
    assert result["frames"] == 0
    assert "valid fresh frames" in result["reason"]


def test_isolated_dropout_and_speed_spike_are_skipped_without_failing_check():
    dropout = run_at_fps(5.0, dropout_index=7)
    spike = run_at_fps(5.0, spike_index=7)
    assert dropout.state == "PASSED" and dropout.count >= MIN_VALID_FRAMES
    assert spike.state == "PASSED" and spike.count >= MIN_VALID_FRAMES


def test_timestamp_gap_and_duplicate_are_not_counted_or_interpolated():
    check = PositioningCheck()
    check.begin()
    times = [index * 0.2 for index in range(6)] + [2.0 + index * 0.2 for index in range(31)]
    for seconds in times:
        check.observe(
            seconds, leg(), KneeAngleResult(moving_angle(seconds), True, 0.9, "ok"), 0.3
        )
        if seconds == 0.8:
            before = check.count
            check.observe(seconds, leg(), KneeAngleResult(20.0, True, 0.9, "ok"), 0.3)
            assert check.count == before
        if check.state != "CHECKING":
            break
    assert check.state == "PASSED"
    assert check.minimum > 100.0


def test_persistently_invalid_tracking_retries_at_eight_seconds():
    check = PositioningCheck()
    check.begin()
    for index in range(41):
        check.observe(
            index * 0.2, None, KneeAngleResult(None, False, 0.0, "pose_missing"), 0.3
        )
    assert check.state == "RETRY"
    assert check.count == 0
    assert "valid fresh frames" in check.reason


@pytest.mark.parametrize("index,value", [(1, "quality"), (2, "clockwise_90"),
                                         (3, "left"), (4, "left"), (5, 1280)])
def test_pass_resets_when_setup_or_tracked_leg_changes(index, value):
    check = PositioningCheck()
    sweep(check)
    context = list(check.context)
    context[index] = value
    check.set_context(tuple(context))
    assert check.state == "READY"
    assert check.manual_side_on is False


def test_passed_check_warns_without_cancelling_or_erasing_pass():
    check = PositioningCheck()
    sweep(check)
    check.observe(5, replace(leg(), hip=Landmark(.99, .22, 1, 1)), KneeAngleResult(150, True, 1, "ok"), .3)
    assert check.state == "PASSED"
    assert "outside safe area" in check.warning
    check.observe(6, None, KneeAngleResult(None, False, 0, "pose_missing"), .3)
    assert "invalid" in check.warning
    assert check.state == "PASSED"


def test_stale_duplicate_timestamps_and_too_short_check_cannot_pass():
    check = PositioningCheck()
    check.begin(True)
    with pytest.raises(RuntimeError):
        PositioningCheck().finish(0)
    check.observe(0, leg(), KneeAngleResult(175, True, 0.9, "ok"), 0.3)
    for _ in range(20):
        check.observe(0, leg(), KneeAngleResult(120, True, 0.9, "ok"), 0.3)
    assert check.count == 1 and check.state == "CHECKING"
    assert check.finish(MAX_CHECK_DURATION_S)["state"] == "RETRY"
    check = PositioningCheck()
    sweep(check)
    check.reset("restart")
    assert check.state != "PASSED"


@pytest.mark.parametrize("rotation", ["none", "clockwise_90", "counterclockwise_90", "rotate_180"])
def test_guides_align_with_the_rotated_inference_frame(rotation, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from computer.vision_imu_calibration.overlay import draw_leg_overlay, _pixel
    from computer.vision_imu_calibration.runtime import VisionRuntime
    from computer.vision_imu_calibration.tuning import FrameRotation
    frame = VisionRuntime._rotate_frame(np.zeros((240, 400, 3), dtype=np.uint8), FrameRotation(rotation), cv2)
    h, w = frame.shape[:2]
    result = draw_leg_overlay(frame, None, KneeAngleResult(None, False, 0, "pose_missing"),
                              inference_fps=0, source_description="test",
                              positioning={"enabled": True, "state": "READY"})
    x, y = _pixel(SAFE_AREA[0], SAFE_AREA[1], w, h)
    assert np.any(result[y, x])
    # The removed fixed hip cross must not paint an arbitrary central target.
    center_x, center_y = _pixel(.5, .22, w, h)
    assert not np.any(result[center_y, center_x])
    assert _pixel(.5, .22, w, h) == (round(.5 * (w - 1)), round(.22 * (h - 1)))


def test_blinded_overlay_keeps_skeleton_and_arc_but_removes_only_angle_text(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from computer.vision_imu_calibration.overlay import draw_leg_overlay
    labels = []
    original = cv2.putText
    def spy(frame, text, *args, **kwargs):
        labels.append(text)
        return original(frame, text, *args, **kwargs)
    monkeypatch.setattr(cv2, "putText", spy)
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    output = draw_leg_overlay(frame, leg(), KneeAngleResult(123, True, .9, "ok"),
                              inference_fps=15, source_description="test", hide_angle=True)
    assert "TRACKING VALID" in labels and "RIGHT" in labels
    assert not any("deg" in text for text in labels)
    assert np.any(output[100:])
