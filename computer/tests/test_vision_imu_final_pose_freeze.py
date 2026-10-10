"""Final UNO Q selected-configuration regression checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from computer.vision_imu_calibration.models import LegSide
from computer.vision_imu_calibration.runtime import VisionRuntime
from computer.vision_imu_calibration.tuning import RuntimeTuning


class Camera:
    description = "fake"
    width = 640
    height = 480
    fps = 30
    camera_index = 0
    active_camera_index = 0

class Estimator:
    description = "fake"


def test_final_uno_q_frozen_values_ignore_stale_local_preset(tmp_path: Path):
    folder = tmp_path / 'calibration'
    folder.mkdir()
    (folder / 'pose_tuning_preset_v1.json').write_text(json.dumps({
        'schema_version': 1,
        'pose_tuning': {
            'confidence_threshold': .41,
            'landmark_smoothing': .55,
            'angle_smoothing': .35,
            'min_segment_length': .045,
            'max_angle_jump_deg': 70.0,
        }
    }))
    r = VisionRuntime(Camera(), Estimator(), leg_side=LegSide.LEFT,
                      pose_filters_locked=True, recording_root=tmp_path/'recordings',
                      calibration_root=folder)
    values=r.tuning()
    assert values['leg_side'] == 'left'
    for field, target in {
        'confidence_threshold':.29,'landmark_smoothing':.30,
        'angle_smoothing':.70,'min_segment_length':.025,
        'max_angle_jump_deg':45.0}.items():
        assert values[field] == pytest.approx(target)
    assert r.pose_preset_status()['locked'] is True
    assert r.pose_preset_status()['loaded'] is False
    assert r.overlay_stroke_status() == {'stroke_scale':.7}
    # No old preset was deleted or overwritten.
    assert json.loads((folder / 'pose_tuning_preset_v1.json').read_text())['pose_tuning']['angle_smoothing'] == .35
    # Leg choice is a valid physical-setup change.
    assert r.update_tuning({'leg_side':'right'})['leg_side'] == 'right'
    assert r.tuning()['angle_smoothing'] == pytest.approx(.70)
    # Reject accidental override through older API clients.
    with pytest.raises(ValueError, match='fixed'):
        r.update_tuning({'landmark_smoothing':.5})
    with pytest.raises(ValueError, match='fixed'):
        r.save_tuning_preset({})
    with pytest.raises(ValueError, match='fixed'):
        r.set_overlay_stroke_scale(.9)
    assert r.tuning()['landmark_smoothing'] == pytest.approx(.30)


def test_pc_mode_retains_backward_compatible_live_tuning(tmp_path):
    r = VisionRuntime(Camera(), Estimator(), calibration_root=tmp_path/'calibration',
                      recording_root=tmp_path/'recordings')
    assert r.pose_preset_status()['locked'] is False
    assert r.update_tuning({'landmark_smoothing':.58})['landmark_smoothing'] == pytest.approx(.58)


def test_applab_launch_pins_lightning_four_threads_left_and_lock():
    project=Path(__file__).resolve().parents[2]
    contents=(project/'standalone_tools/vision_imu_calibration/uno_q_phase2_app_lab_template/python/main.py').read_text()
    for option, value in (('"--pose-model"','"lightning"'), ('"--inference-threads"','"4"'), ('"--leg"','"left"')):
        assert contents.index(option) < contents.index(value, contents.index(option))
    assert '"--final-pose-lock"' in contents
