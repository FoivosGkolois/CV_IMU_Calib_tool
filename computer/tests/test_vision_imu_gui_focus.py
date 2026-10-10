from __future__ import annotations

import re

from computer.vision_imu_calibration.tuning import FrameRotation, RuntimeTuning
from computer.vision_imu_calibration.web_app import _PAGE


def test_workflow_has_five_ordered_single_open_primary_steps() -> None:
    step_ids = re.findall(
        r'<details id="([^"]+)" class="card workflow-step[^"]*"[^>]*data-workflow-step[^>]*>',
        _PAGE,
    )
    assert step_ids == [
        "cameraSetupStep",
        "positionStep",
        "calibrationStep",
        "validationStep",
        "resultsStep",
    ]
    assert _PAGE.count("data-workflow-step") == 5
    assert 'id="referencePanel" class="card auxiliary-panel" data-focus-panel' in _PAGE
    assert _PAGE.count("data-focus-panel open") == 1

    accordion = _PAGE.split("let accordionSync=false;", 1)[1].split(
        "function updateSliderLabels", 1
    )[0]
    assert "if(other!==step) other.open=false" in accordion
    assert "cancel" not in accordion.lower()
    assert ".value=" not in accordion


def test_camera_remains_prominent_and_uncropped_outside_workflow() -> None:
    camera = _PAGE.index('<section class="visual-column"')
    workflow = _PAGE.index('<aside class="control-panel">')
    assert camera < workflow
    assert "grid-template-columns:minmax(0,58fr) minmax(380px,42fr)" in _PAGE
    assert ".visual-column { position:sticky" in _PAGE
    assert ".video-shell img { width:100%; height:100%; object-fit:contain" in _PAGE
    assert "videoShell.style.aspectRatio=String(naturalWidth/naturalHeight)" in _PAGE
    assert "max-height:88vh" in _PAGE
    assert "@media (max-width:1100px)" in _PAGE
    assert "@media (max-width:760px)" in _PAGE
    assert "position:fixed; inset:0" not in _PAGE
    assert 'aria-modal="true"' not in _PAGE


def test_five_pose_sliders_live_only_under_advanced_and_defaults_are_unchanged() -> None:
    advanced = _PAGE.split("Advanced / Diagnostics", 1)[1].split("</details>", 1)[0]
    for slider_id in (
        "confidenceThreshold",
        "landmarkSmoothing",
        "angleSmoothing",
        "segmentLength",
        "angleJump",
    ):
        assert _PAGE.count(f'id="{slider_id}"') == 1
        assert f'id="{slider_id}"' in advanced
    assert _PAGE.count('type="range"') == 5
    assert 'id="applyPoseSettings"' in advanced
    assert 'id="savePosePreset"' in advanced
    assert "fetch('/api/tuning/preset'" in _PAGE
    assert "const payload={leg_side:legSide.value,frame_rotation:frameRotation.value}" in _PAGE
    assert "fetch('/api/tuning'" in _PAGE

    defaults = RuntimeTuning()
    assert defaults.confidence_threshold == 0.30
    assert defaults.landmark_smoothing == 0.55
    assert defaults.angle_smoothing == 0.35
    assert defaults.min_segment_length == 0.025
    assert defaults.max_angle_jump_deg == 45.0
    assert defaults.frame_rotation is FrameRotation.COUNTERCLOCKWISE_90

    changed = RuntimeTuning.from_mapping(
        {"leg_side": "left", "frame_rotation": "none"}, current=defaults
    )
    assert changed.confidence_threshold == defaults.confidence_threshold
    assert changed.landmark_smoothing == defaults.landmark_smoothing
    assert changed.angle_smoothing == defaults.angle_smoothing
    assert changed.min_segment_length == defaults.min_segment_length
    assert changed.max_angle_jump_deg == defaults.max_angle_jump_deg


def test_focused_controls_keep_ids_endpoints_blinding_and_exports() -> None:
    required_ids = {
        "referenceSetup",
        "referenceMeasured",
        "referenceConfirm",
        "referenceCapture",
        "calibrationModal",
        "calibrationCountdown",
        "startStaticTest",
        "startDynamicTest",
        "startRecording",
        "stopRecording",
        "heldoutUnchangedSetup",
        "validateHeldout",
        "referenceExport",
        "recordingPath",
        "calibrationPath",
        "heldoutPath",
    }
    page_ids = re.findall(r'\bid="([^"]+)"', _PAGE)
    assert len(page_ids) == len(set(page_ids))
    assert required_ids <= set(page_ids)
    for endpoint in (
        "/api/goniometer",
        "/api/calibration/start",
        "/api/recording/start",
        "/api/recording/stop",
        "/api/calibration/validate-heldout",
        "/exports/recordings/",
        "/exports/calibration/",
    ):
        assert endpoint in _PAGE
    assert "setReferenceBlind(true)" in _PAGE
    assert "Capture 3-second hold" in _PAGE
    assert "class=\"inline-guidance\" hidden" in _PAGE
    assert 'id="calibrationSetup"' not in _PAGE
    assert 'id="sideOn"' not in _PAGE
    assert "positionAction('check')" in _PAGE
    assert "positionAction('start')" not in _PAGE
    assert "positionAction('finish')" not in _PAGE


def test_every_literal_javascript_dom_lookup_has_a_matching_element() -> None:
    page_ids = set(re.findall(r'\bid="([^"]+)"', _PAGE))
    looked_up = set(re.findall(r"getElementById\('([^']+)'\)", _PAGE))
    assert looked_up <= page_ids


def test_advanced_diagnostics_contains_nonessential_runtime_controls() -> None:
    advanced = _PAGE.split("Advanced / Diagnostics", 1)[1].split("</details>", 1)[0]
    for element_id in (
        "imuSerialControls",
        "imuRate",
        "imuSensors",
        "imuHealth",
        "angleDifference",
        "imuSpeed",
        "segmentAngles",
        "reloadImuFilter",
        "backend",
        "source",
        "frameCount",
        "startGeneralRecording",
        "stopGeneralRecording",
    ):
        assert f'id="{element_id}"' in advanced
    assert "Advanced / Diagnostics</span></summary>" in _PAGE
