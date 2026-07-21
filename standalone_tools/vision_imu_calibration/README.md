# Standalone Vision–IMU calibration hardware

This folder contains the temporary acquisition firmware used by the standalone Vision–IMU tool. The sketches initialize only the two LSM6DSOX IMUs and the USB serial link. They contain no motor, encoder, EMG, FSR, assistance, or safety-control code.

Both firmware variants preserve the same `IMUCAL v1` protocol, field order, raw units, timestamps, sample index, and 100 Hz target.

## Current UNO R4 baseline

Upload:

```text
arduino/uno_r4_dual_imu_stream/uno_r4_dual_imu_stream.ino
```

Use it with:

```text
11_RUN_VISION_IMU_SYNC_PC.bat
```

## UNO Q migration — Phase 1

Phase 1 moves only the deterministic dual-IMU acquisition sketch to the UNO Q STM32U585 MCU. Python, MoveNet, the web GUI, synchronization, recording, and calibration still run on the Windows PC.

Upload with Arduino IDE 2:

```text
arduino/uno_q_dual_imu_stream/uno_q_dual_imu_stream.ino
```

Required setup:

```text
Board: Arduino UNO Q
Arduino UNO Q Board platform: 0.55.0 or newer
IMUs: Qwiic / Wire1
Thigh address: 0x6A
Shank address: 0x6B
```

Do not run the sketch as an Arduino App Lab App during Phase 1. Upload the MCU-only sketch with Arduino IDE, close Serial Monitor, and then run:

```text
12_TEST_UNO_Q_PHASE1_PC.bat
```

The validator performs a 15-second capture and checks:

- both IMUs ready,
- fixed `IMUCAL v1` parsing,
- approximately 100 Hz acquisition,
- MCU timestamp period,
- sample-index continuity,
- zero parse errors and host queue drops,
- valid and physically plausible accelerometer samples.

It creates exactly one result ZIP inside a dedicated folder under:

```text
diagnostic_logs\uno_q_phase1_<timestamp>\
```

After the validation passes, the existing GUI can use the UNO Q COM port without any Python or protocol change:

```text
11_RUN_VISION_IMU_SYNC_PC.bat
```

The standalone acquisition path remains separate from the production exoskeleton firmware so the migration cannot modify the established production telemetry or control lifecycle.
