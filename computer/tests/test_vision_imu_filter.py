from __future__ import annotations

from pathlib import Path

import pytest

from computer.vision_imu_calibration.imu_filter import (
    KneeAngleFilterMirror,
    KneeAngleFilterParameters,
    resolve_filter_config_path,
)
from computer.vision_imu_calibration.imu_serial import (
    ACCEL_G_PER_LSB,
    GYRO_DPS_PER_LSB,
    ImuCalibrationSample,
)


def sample(
    *,
    index: int = 1,
    timestamp_us: int = 10_000,
    thigh_ok: bool = True,
    shank_ok: bool = True,
    thigh_acc: tuple[float, float, float] = (0.0, 0.0, 1.0),
    shank_acc: tuple[float, float, float] = (0.0, 0.0, 1.0),
    thigh_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
    shank_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ImuCalibrationSample:
    accel = lambda value: int(round(value / ACCEL_G_PER_LSB))
    gyro = lambda value: int(round(value / GYRO_DPS_PER_LSB))
    return ImuCalibrationSample(
        host_monotonic_ns=timestamp_us * 1000,
        schema_version=1,
        sample_index=index,
        mcu_timestamp_us=timestamp_us,
        thigh_ok=thigh_ok,
        shank_ok=shank_ok,
        thigh_ax_raw=accel(thigh_acc[0]),
        thigh_ay_raw=accel(thigh_acc[1]),
        thigh_az_raw=accel(thigh_acc[2]),
        thigh_gx_raw=gyro(thigh_gyro[0]),
        thigh_gy_raw=gyro(thigh_gyro[1]),
        thigh_gz_raw=gyro(thigh_gyro[2]),
        shank_ax_raw=accel(shank_acc[0]),
        shank_ay_raw=accel(shank_acc[1]),
        shank_az_raw=accel(shank_acc[2]),
        shank_gx_raw=gyro(shank_gyro[0]),
        shank_gy_raw=gyro(shank_gyro[1]),
        shank_gz_raw=gyro(shank_gyro[2]),
    )


def test_loads_current_stm32_knee_angle_parameters() -> None:
    root = Path(__file__).resolve().parents[2]
    path = resolve_filter_config_path(working_directory=root)
    params = KneeAngleFilterParameters.load_yaml(path)
    assert params.rotation_axis == 0
    assert params.gyro_rate_sign == pytest.approx(1.0)
    assert params.acc_axis_1_sign == pytest.approx(-1.0)
    assert params.relative_angle_sign == pytest.approx(-1.0)
    assert params.alpha_stable == pytest.approx(0.9)
    assert params.alpha_dynamic == pytest.approx(0.975)


def test_standalone_config_falls_back_to_bundled_snapshot(tmp_path: Path) -> None:
    path = resolve_filter_config_path(working_directory=tmp_path)
    params = KneeAngleFilterParameters.load_yaml(path)
    assert path.name == "default_knee_filter.yaml"
    assert params.alpha_dynamic == pytest.approx(0.975)


def test_explicit_missing_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_filter_config_path(tmp_path / "missing.yaml")


def test_static_straight_pose_returns_one_hundred_eighty_degrees() -> None:
    mirror = KneeAngleFilterMirror(KneeAngleFilterParameters())
    result = mirror.update(sample())
    assert result.valid
    assert result.knee_angle_deg == pytest.approx(180.0, abs=0.2)
    assert result.knee_speed_deg_s == pytest.approx(0.0)


def test_known_relative_accelerometer_pose_returns_ninety_degrees() -> None:
    mirror = KneeAngleFilterMirror(KneeAngleFilterParameters())
    result = mirror.update(sample(shank_acc=(0.0, -1.0, 0.0)))
    assert result.valid
    assert result.knee_angle_deg == pytest.approx(90.0, abs=0.2)


def test_gyro_integration_updates_the_new_angle_convention() -> None:
    params = KneeAngleFilterParameters(
        alpha_stable=1.0,
        alpha_dynamic=1.0,
        gyro_deadband_deg_s=0.0,
    )
    mirror = KneeAngleFilterMirror(params)
    mirror.update(sample(index=1, timestamp_us=10_000))
    result = mirror.update(
        sample(index=2, timestamp_us=20_000, shank_gyro=(100.0, 0.0, 0.0))
    )
    assert result.valid
    assert result.knee_angle_deg == pytest.approx(179.0, abs=0.02)
    assert result.knee_speed_deg_s == pytest.approx(-100.0, abs=0.02)


def test_invalid_sensor_frame_resets_filter_state() -> None:
    mirror = KneeAngleFilterMirror(KneeAngleFilterParameters())
    assert mirror.update(sample()).valid
    invalid = mirror.update(sample(index=2, timestamp_us=20_000, thigh_ok=False))
    assert not invalid.valid
    assert invalid.reason == "imu_not_ready"
    restored = mirror.update(sample(index=3, timestamp_us=30_000, shank_acc=(0.0, -1.0, 0.0)))
    assert restored.knee_angle_deg == pytest.approx(90.0, abs=0.2)
