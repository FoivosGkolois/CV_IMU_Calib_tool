"""Regression tests for the approved guided, blinded goniometer and pose-tuning UI."""
from __future__ import annotations

from dataclasses import replace

import pytest

from computer.vision_imu_calibration.reference_validation import GONIOMETER_TARGETS
from computer.vision_imu_calibration.runtime import VisionRuntime
from computer.vision_imu_calibration.pose_tuning_page import TUNING_PAGE
from computer.vision_imu_calibration.web_app import _PAGE


class FakeCamera:
    description = "test-camera"
    width = 640
    height = 480
    fps = 30
    camera_index = 0
    active_camera_index = 0


class FakeEstimator:
    description = "test-pose"


def make_runtime(tmp_path):
    runtime = VisionRuntime(FakeCamera(), FakeEstimator(), recording_root=tmp_path / "recordings",
                            calibration_root=tmp_path / "calibration")
    runtime._status = replace(runtime._status, state="RUNNING", tracked_leg="right",
                               tracking_valid=True)
    return runtime


def test_guided_reference_starts_blinded_at_90_with_auto_reference(tmp_path):
    r = make_runtime(tmp_path)
    result = r.reference_action("start_session", {})
    assert result["hide_angle"]
    assert result["guided"]["target_deg"] == 90
    assert result["guided"]["repeat"] == 1
    assert result["guided"]["total"] == 10
    assert r._reference_inputs["reference_angle_deg"] == 90
    assert r._reference_inputs["reference_method"] == "operator_positioned_with_physical_goniometer_at_requested_reference"
    assert r.status()["visual_knee_angle_deg"] is None
    assert not r.reference_status()["guided"]["captures"]
    assert r.reference_action("cancel", {})["guided"] is None


def test_guided_reference_rejects_next_before_capture_and_profile_change(tmp_path):
    r = make_runtime(tmp_path)
    r.reference_action("start_session", {})
    with pytest.raises(RuntimeError, match="Finish or retry"):
        r.reference_action("next", {})
    r._reference.update(state="MEASURED", hide_angle=False)
    r.update_tuning({"confidence_threshold": .42})
    with pytest.raises(RuntimeError, match="profile changed"):
        r.reference_action("next", {})


def test_gui_has_no_legacy_tracking_gate_or_manual_goniometer_forms():
    assert "positionAction('check')" not in _PAGE
    assert 'id="positionStep"' not in _PAGE
    for old in ('referenceMeasured', 'referenceSetup', 'referenceReuse', 'savedReference', 'unchangedSetup'):
        assert f'id="{old}"' not in _PAGE
    assert 'id="referenceProgressBar"' in _PAGE
    assert "referenceAction('start_session')" in _PAGE
    assert "referenceAction('next')" in _PAGE
    assert "referenceAction('capture')" in _PAGE
    assert GONIOMETER_TARGETS == (90, 120, 140, 160, 175)
    assert "target+'°'" in _PAGE  # The angle is shown dynamically, not as static help text.
    for landmark in ('hip', 'knee', 'ankle'):
        assert f'id="{landmark}Confidence"' in _PAGE
    assert 'id="confidence"' not in _PAGE
    assert 'id="liveChartPanel"' in _PAGE


def test_standalone_tuning_only_uses_supported_host_parameter_keys():
    for key in ('confidence_threshold', 'landmark_smoothing', 'angle_smoothing',
                'min_segment_length', 'max_angle_jump_deg'):
        assert key in TUNING_PAGE
    for url in ('/api/pose-tuning/history', '/api/tuning', '/api/tuning/preset', '/api/status'):
        assert url in TUNING_PAGE
    assert 'Save selected values as default' in TUNING_PAGE
    assert 'Download test evidence JSON' in TUNING_PAGE
    assert 'No setting is committed automatically' in TUNING_PAGE


def test_tuning_history_keeps_native_observation_and_leg_scores(tmp_path):
    r=make_runtime(tmp_path)
    sample={"frame_index":8,"host_monotonic_ns":__import__('time').perf_counter_ns(),
            "tracking_valid":True,"raw_angle_deg":110.0,
            "filtered_angle_deg":111.0,"hip":.8,"knee":.6,"ankle":.9}
    r._pose_tuning_history.append(sample)
    found=r.pose_tuning_history(seconds=10)["rows"]
    assert len(found)==1 and found[0]["knee"]==.6
    found[0]["hip"]=.01
    assert r._pose_tuning_history[0]["hip"]==.8  # no mutable alias
    with pytest.raises(ValueError):
        r.pose_tuning_history(seconds=40)
    r._reference["hide_angle"] = True
    assert r.pose_tuning_history(seconds=10)["rows"][0]["raw_angle_deg"] is None


def test_overlay_stroke_is_session_only_and_strictly_validated(tmp_path):
    r = make_runtime(tmp_path)
    assert r._overlay_stroke_scale == 1.0
    assert r.set_overlay_stroke_scale(0.4) == {"stroke_scale": 0.4}
    assert r._overlay_stroke_scale == 0.4
    assert r.overlay_stroke_status() == {"stroke_scale": 0.4}
    for rejected in (0, 1.3, float("nan"), float("inf"), True, "unknown", None):
        with pytest.raises(ValueError):
            r.set_overlay_stroke_scale(rejected)
    # Pose tuning and calibration profile are untouched by this presentation-only setting.
    assert r._tuning.confidence_threshold == 0.30
    assert "stroke_scale" not in r._vision_setup()["tuning"]


def test_goniometer_prominent_capture_display_and_no_redundant_help():
    for part in ('id="referencePosition"', 'id="referenceRepeat"', 'id="referenceTarget"',
                 'id="referenceInstruction"', 'id="referenceProgressBar"',
                 'id="referenceResultsPanel"'):
        assert part in _PAGE
    assert 'Goniometer placement and 2-D camera geometry introduce uncertainty' not in _PAGE
    assert 'Position ${guide.index+1} / 10' not in _PAGE
    assert 'id="overlayStroke"' in _PAGE
    assert 'id="overlayStrokeValue"' in _PAGE
    assert "fetch('/api/overlay/stroke'" in _PAGE
