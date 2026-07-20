from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from computer.vision_imu_calibration.imu_filter import (
    KneeAngleFilterMirror,
    KneeAngleFilterParameters,
)
from computer.vision_imu_calibration.imu_serial import (
    ACCEL_G_PER_LSB,
    GYRO_DPS_PER_LSB,
    ImuCalibrationSample,
)


def test_python_filter_matches_current_cpp_implementation(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("desktop C++ compiler unavailable")

    root = Path(__file__).resolve().parents[2]
    src = root / "device" / "stm32_firmware" / "libraries" / "ExoStm32Core" / "src"
    if not (src / "knee_angle.cpp").is_file() or not (src / "knee_angle.h").is_file():
        pytest.skip("STM32 source tree is unavailable in this standalone calibration-tool copy")
    (tmp_path / "Arduino.h").write_text(
        "#pragma once\n#include <cstdint>\n#include <cstddef>\n#define A0 0\n#define A1 1\n#define A2 2\n#define A3 3\nusing std::uint8_t; using std::int8_t; using std::uint16_t; using std::int16_t; using std::uint32_t; class Stream {};\n",
        encoding="utf-8",
    )
    (tmp_path / "Wire.h").write_text(
        "#pragma once\n#include <cstdint>\nclass TwoWire {};\nstatic TwoWire Wire; static TwoWire Wire1;\n",
        encoding="utf-8",
    )
    harness = tmp_path / "parity.cpp"
    harness.write_text(
        r'''
#include <iomanip>
#include <iostream>
#include "knee_angle.h"
using namespace exo;

void setImu(ImuMeasurement& imu, int ax, int ay, int az, int gx, int gy, int gz) {
  imu.ok = true;
  imu.ax_g = ax * 0.000061f; imu.ay_g = ay * 0.000061f; imu.az_g = az * 0.000061f;
  imu.gx_dps = gx * 0.00875f; imu.gy_dps = gy * 0.00875f; imu.gz_dps = gz * 0.00875f;
}
int main() {
  RuntimeParameters p;
  p.knee_imu_rotation_axis = 0;
  p.knee_imu_gyro_rate_sign = 1.0f;
  p.knee_imu_acc_axis_1_sign = -1.0f;
  p.knee_imu_acc_axis_2_sign = 1.0f;
  p.assist_imu_relative_angle_sign = 1.0f;
  p.knee_imu_filter_alpha_stable = 0.90f;
  p.knee_imu_filter_alpha_dynamic = 0.975f;
  p.knee_imu_acc_full_trust_deviation_g = 0.06f;
  p.knee_imu_acc_no_trust_deviation_g = 0.30f;
  p.knee_imu_max_accel_correction_deg = 10.0f;
  p.knee_imu_gyro_deadband_deg_s = 1.0f;
  resetKneeAngleProcessor();
  SensorMeasurements s;
  const int trace[6][12] = {
    {0,0,16393,0,0,0, 0,0,16393,0,0,0},
    {0,0,16393,0,0,0, 0,-1000,16362,1143,0,0},
    {0,0,16393,0,0,0, 0,-3000,16116,1143,0,0},
    {0,0,16393,0,0,0, 0,-6000,15256,0,0,0},
    {0,0,16393,0,0,0, 0,-9000,13699,-1143,0,0},
    {0,0,16393,0,0,0, 0,-12000,11162,-1143,0,0}
  };
  std::cout << std::setprecision(9);
  for (int i=0;i<6;i++) {
    s.sample_index = i+1; s.timestamp_us = (i+1)*10000;
    setImu(s.thigh_imu, trace[i][0],trace[i][1],trace[i][2],trace[i][3],trace[i][4],trace[i][5]);
    setImu(s.shank_imu, trace[i][6],trace[i][7],trace[i][8],trace[i][9],trace[i][10],trace[i][11]);
    auto e = estimateCalibratedKneeAngle(s,p);
    std::cout << e.knee_angle_deg << "," << e.knee_speed_deg_s << "," << e.thigh_angle_deg << "," << e.shank_angle_deg << "\n";
  }
}
''',
        encoding="utf-8",
    )
    executable = tmp_path / "parity"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{tmp_path}",
            f"-I{src}",
            str(harness),
            str(src / "knee_angle.cpp"),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    cpp_rows = [
        [float(value) for value in line.split(",")]
        for line in subprocess.check_output([str(executable)], text=True).strip().splitlines()
    ]

    params = KneeAngleFilterParameters()
    mirror = KneeAngleFilterMirror(params)
    trace = [
        (0, 0, 16393, 0, 0, 0, 0, 0, 16393, 0, 0, 0),
        (0, 0, 16393, 0, 0, 0, 0, -1000, 16362, 1143, 0, 0),
        (0, 0, 16393, 0, 0, 0, 0, -3000, 16116, 1143, 0, 0),
        (0, 0, 16393, 0, 0, 0, 0, -6000, 15256, 0, 0, 0),
        (0, 0, 16393, 0, 0, 0, 0, -9000, 13699, -1143, 0, 0),
        (0, 0, 16393, 0, 0, 0, 0, -12000, 11162, -1143, 0, 0),
    ]
    py_rows = []
    for index, values in enumerate(trace, start=1):
        sample = ImuCalibrationSample(
            host_monotonic_ns=index * 10_000_000,
            schema_version=1,
            sample_index=index,
            mcu_timestamp_us=index * 10_000,
            thigh_ok=True,
            shank_ok=True,
            thigh_ax_raw=values[0], thigh_ay_raw=values[1], thigh_az_raw=values[2],
            thigh_gx_raw=values[3], thigh_gy_raw=values[4], thigh_gz_raw=values[5],
            shank_ax_raw=values[6], shank_ay_raw=values[7], shank_az_raw=values[8],
            shank_gx_raw=values[9], shank_gy_raw=values[10], shank_gz_raw=values[11],
        )
        estimate = mirror.update(sample)
        py_rows.append([
            estimate.knee_angle_deg,
            estimate.knee_speed_deg_s,
            estimate.thigh_angle_deg,
            estimate.shank_angle_deg,
        ])

    for cpp, python in zip(cpp_rows, py_rows, strict=True):
        expected_python = [180.0 - cpp[0], cpp[1], cpp[2], cpp[3]]
        assert python == pytest.approx(expected_python, abs=2e-5)
