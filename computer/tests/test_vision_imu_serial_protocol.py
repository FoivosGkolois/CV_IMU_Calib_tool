from __future__ import annotations

import pytest

from computer.vision_imu_calibration.imu_serial import (
    ACCEL_G_PER_LSB,
    GYRO_DPS_PER_LSB,
    ImuSerialProtocol,
)


def valid_frame() -> str:
    return (
        "IMUCAL,1,42,123456,1,1,"
        "100,-200,300,10,-20,30,"
        "400,-500,600,40,-50,60"
    )


def test_parse_fixed_imu_calibration_frame() -> None:
    sample = ImuSerialProtocol.parse_sample(valid_frame(), host_monotonic_ns=999)
    assert sample.sample_index == 42
    assert sample.mcu_timestamp_us == 123456
    assert sample.thigh_ok and sample.shank_ok
    assert sample.thigh_ay_raw == -200
    assert sample.shank_gz_raw == 60
    record = sample.to_record()
    assert record["thigh_ax_g"] == pytest.approx(100 * ACCEL_G_PER_LSB)
    assert record["shank_gz_dps"] == pytest.approx(60 * GYRO_DPS_PER_LSB)


def test_reject_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="field count"):
        ImuSerialProtocol.parse_sample("IMUCAL,1,2", host_monotonic_ns=1)


def test_reject_wrong_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported IMUCAL schema"):
        ImuSerialProtocol.parse_sample(valid_frame().replace("IMUCAL,1,", "IMUCAL,2,"), host_monotonic_ns=1)


def test_reject_non_integer_field() -> None:
    with pytest.raises(ValueError, match="non-integer"):
        ImuSerialProtocol.parse_sample(valid_frame().replace(",100,", ",abc,"), host_monotonic_ns=1)


def test_source_counts_sequence_gaps_and_buffers_samples() -> None:
    from computer.vision_imu_calibration.imu_serial import SerialImuSource

    source = SerialImuSource("COM_TEST", queue_size=32)
    first = ImuSerialProtocol.parse_sample(valid_frame(), host_monotonic_ns=1_000_000_000)
    second_line = valid_frame().replace("IMUCAL,1,42,", "IMUCAL,1,45,")
    second = ImuSerialProtocol.parse_sample(second_line, host_monotonic_ns=1_010_000_000)
    source._accept_sample(first)
    source._accept_sample(second)
    status = source.status()
    assert status["sample_count"] == 2
    assert status["sequence_gaps"] == 2
    assert source.get_sample(timeout_s=0.0) == first
    assert source.get_sample(timeout_s=0.0) == second
