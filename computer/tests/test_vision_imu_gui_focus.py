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
        "referencePanel",
        "calibrationStep",
        "validationStep",
        "resultsStep",
    ]
    assert _PAGE.count("data-workflow-step") == 5
    assert 'id="referencePanel" class="card workflow-step" data-workflow-step data-focus-panel' in _PAGE
    numbered = re.findall(r'<span class="step-number">(\d+)</span>', _PAGE)
    assert numbered == ["1", "2", "3", "4", "5"]
    assert _PAGE.count("data-focus-panel open") == 1

    accordion = _PAGE.split("let accordionSync=false;", 1)[1].split(
        "function loadTuning", 1
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
    assert "max-height:100%" in _PAGE
    assert "@media (max-width:1100px)" in _PAGE
    assert "@media (max-width:760px)" in _PAGE
    assert "position:fixed; inset:0" not in _PAGE
    assert 'aria-modal="true"' not in _PAGE


def test_compact_status_and_collapsible_chart_preserve_camera_space() -> None:
    visual = _PAGE.split('<section class="visual-column"', 1)[1].split('</section>', 1)[0]
    controls = _PAGE.split('<aside class="control-panel">', 1)[1]
    assert 'id="angleChart"' not in visual
    assert controls.index('id="liveChartPanel"') < controls.index('id="imuReadySummary"')
    assert '<details id="liveChartPanel" class="chart-card">' in controls
    assert "grid-template-rows:repeat(5,auto)" in _PAGE
    assert "grid-template-rows:minmax(0,1fr) clamp(210px,28vh,290px)" not in _PAGE
    assert "liveChartPanel.addEventListener('toggle'" in _PAGE
    assert "observer.observe(angleChart.parentElement);" in _PAGE


def test_final_pose_settings_are_read_only_and_camera_setup_remains_adjustable() -> None:
    advanced = _PAGE.split("Advanced / Diagnostics", 1)[1].split("</details>", 1)[0]
    for slider_id in ("confidenceThreshold", "landmarkSmoothing", "angleSmoothing",
                      "segmentLength", "angleJump", "overlayStroke"):
        assert f'id="{slider_id}"' not in _PAGE
    assert 'type="range"' not in _PAGE
    assert 'id="applyPoseSettings"' not in advanced
    assert 'id="savePosePreset"' not in advanced
    assert "confidence 0.29" in advanced
    assert "landmark smoothing 0.30" in advanced
    assert "angle smoothing 0.70" in advanced
    assert "minimum segment 0.025" in advanced
    assert "maximum jump 45°" in advanced
    assert "const payload={leg_side:legSide.value,frame_rotation:frameRotation.value}" in _PAGE
    assert "fetch('/api/tuning'" in _PAGE

    defaults = RuntimeTuning()
    assert defaults.confidence_threshold == 0.29
    assert defaults.landmark_smoothing == 0.30
    assert defaults.angle_smoothing == 0.70
    assert defaults.min_segment_length == 0.025
    assert defaults.max_angle_jump_deg == 45.0
    assert defaults.frame_rotation is FrameRotation.COUNTERCLOCKWISE_90

    changed = RuntimeTuning.from_mapping(
        {"leg_side": "left", "frame_rotation": "none"}, current=defaults
    )
    for field in ("confidence_threshold", "landmark_smoothing", "angle_smoothing",
                  "min_segment_length", "max_angle_jump_deg"):
        assert getattr(changed, field) == getattr(defaults, field)


def test_focused_controls_keep_ids_endpoints_blinding_and_exports() -> None:
    required_ids = {
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
    assert "positionAction('check')" not in _PAGE
    assert 'id="positionStep"' not in _PAGE
    assert "referenceAction('start_session')" in _PAGE
    assert "referenceAction('next')" in _PAGE
    assert 'id="referenceMeasured"' not in _PAGE
    assert 'id="referenceConfirm"' not in _PAGE
    assert 'id="savedReference"' not in _PAGE


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


def test_responsive_status_fps_and_step_contrast() -> None:
    visual = _PAGE.split('<section class="visual-column"', 1)[1].split('</section>', 1)[0]
    assert visual.count('class="summary-card') == 5
    assert visual.count('id="fps"') == 1
    assert visual.index('id="ankleConfidence"') < visual.index('class="summary-card fps-card"')
    assert 'id="fps" class="value"' in visual
    assert 'clamp(180px,27%,280px)' in _PAGE
    assert 'repeat(5,minmax(0,1fr))' in _PAGE
    assert '@media (max-width:850px)' in _PAGE
    assert '@media (max-width:760px)' in _PAGE
    assert '@media (max-width:440px)' in _PAGE
    assert '.workflow-step .note' in _PAGE
    assert '.workflow-step .control-help' in _PAGE
    assert 'color:#f1f5ff' in _PAGE
    assert "fps.textContent=Number(s.inference_fps).toFixed(1)+' FPS';" in _PAGE
