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

## UNO Q migration — Phase 2

Phase 2 moves the full host application onto the UNO Q Linux processor:

```text
UNO Q MCU   -> reads both IMUs at 100 Hz
Arduino Bridge -> transfers the same raw sample contract internally
UNO Q Linux -> MoveNet, filtering, synchronization, GUI, recording, and calibration
PC browser  -> only displays and controls the web GUI over Wi-Fi/Ethernet
```

The original Windows + UNO R4 workflow and the Windows + UNO Q Phase 1 workflow remain available.

Build the self-contained App Lab package from Windows:

```text
13_BUILD_UNO_Q_PHASE2_APP.bat
```

This creates:

```text
uno_q_phase2_app_ready.zip
```

Open/import the generated App in Arduino App Lab. The App contains:

- the Phase 2 MCU Bridge sketch,
- the current Python package and MoveNet model,
- Python dependencies,
- the web port definition for port 8765.

After the App starts, open:

```text
http://<UNO-Q-IP>:8765
```

The Phase 2 launcher auto-detects the working UVC capture node instead of assuming
that the webcam is always `/dev/video0`. An explicit override remains available
through `VISION_IMU_CAMERA_INDEX`.

Recordings and calibration outputs are written under:

```text
~/vision_imu_data/
```

Phase 2 intentionally does not add motor control. The MCU remains dedicated to deterministic sensor acquisition, and Linux owns the heavy processing and experiment workflow.
