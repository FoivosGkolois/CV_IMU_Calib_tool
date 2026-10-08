"""Isolated T05a physics-based static IMU candidate (no runtime integration)."""
from dataclasses import replace
import math

import pytest

from computer.vision_imu_calibration.calibration_workflow import (
    ArchiveData,
    _derive_physics_static_parameters,
)
from computer.vision_imu_calibration.imu_filter import KneeAngleFilterParameters


def _rows(
    count: int = 400,
    *,
    relative_deg: float = 15.0,
    thigh_bias: float = 1.5,
    shank_bias: float = -0.8,
):
    """Physically plausible gravity projections around the x rotation axis."""
    result = []
    angle = math.radians(relative_deg)
    for index in range(count):
        small_noise = (-0.1, 0.0, 0.1)[index % 3]
        result.append({
            "thigh_ax_g": "0", "thigh_ay_g": "0", "thigh_az_g": "1",
            "shank_ax_g": "0", "shank_ay_g": str(-math.sin(angle)),
            "shank_az_g": str(math.cos(angle)),
            "thigh_gx_dps": str(thigh_bias + small_noise),
            "shank_gx_dps": str(shank_bias + small_noise),
        })
    return result


def test_static_physics_uses_only_imu_not_camera_and_preserves_unrelated_parameters():
    base = replace(KneeAngleFilterParameters(), alpha_dynamic=0.91, alpha_stable=0.83)
    data = ArchiveData(raw_rows=_rows(), sync_rows=[{"visual_angle_deg": "999"}])
    result = _derive_physics_static_parameters(data, base)
    calibrated = result["parameters"]
    assert result["used_visual_reference"] is False
    assert result["physical_reference_condition"] == "assumed_full_extension"
    assert result["sample_count"] == 400
    assert calibrated.straight_leg_offset_deg == pytest.approx(15.0, abs=1e-9)
    assert calibrated.thigh_gyro_bias_deg_s == pytest.approx(1.5, abs=1e-9)
    assert calibrated.shank_gyro_bias_deg_s == pytest.approx(-0.8, abs=1e-9)
    assert calibrated.gyro_deadband_deg_s == pytest.approx(0.44478, abs=1e-8)
    assert calibrated.alpha_dynamic == base.alpha_dynamic
    assert calibrated.alpha_stable == base.alpha_stable
    assert base.straight_leg_offset_deg == 0.0  # never mutate the input


def test_no_camera_data_yields_same_physics_result():
    raw = _rows(relative_deg=22.0)
    without = _derive_physics_static_parameters(ArchiveData(raw, []), KneeAngleFilterParameters())
    with_camera = _derive_physics_static_parameters(ArchiveData(raw, [{"junk": "1"}]), KneeAngleFilterParameters())
    assert without["parameters"] == with_camera["parameters"]
    assert without["straight_leg_offset_deg"] == pytest.approx(22.0)


def test_static_physics_honours_configured_axes_and_offsets():
    base = replace(
        KneeAngleFilterParameters(), rotation_axis=2,
        thigh_offset_deg=10.0, shank_offset_deg=-5.0,
        relative_angle_sign=-1.0, gyro_rate_sign=-1.0,
    )
    # With axis z the atan2 inputs are (-ay, ax): shank tilt is 20 degrees.
    raw = [{
        "thigh_ax_g": "1", "thigh_ay_g": "0", "thigh_az_g": "0",
        "shank_ax_g": str(math.cos(math.radians(20))),
        "shank_ay_g": str(-math.sin(math.radians(20))), "shank_az_g": "0",
        "thigh_gz_dps": "-0.6", "shank_gz_dps": "1.1",
    } for _ in range(5)]
    result = _derive_physics_static_parameters(ArchiveData(raw, []), base)
    # Relative: (20 - -5) - (0 - 10) = +35; relative_angle_sign = -1.
    assert result["straight_leg_offset_deg"] == pytest.approx(-35.0)
    assert result["thigh_gyro_bias_deg_s"] == pytest.approx(0.6)
    assert result["shank_gyro_bias_deg_s"] == pytest.approx(-1.1)
    assert result["gyro_deadband_deg_s"] == pytest.approx(0.05)


def test_empty_static_archive_is_rejected():
    with pytest.raises(RuntimeError, match="no raw IMU samples"):
        _derive_physics_static_parameters(ArchiveData([], []), KneeAngleFilterParameters())
