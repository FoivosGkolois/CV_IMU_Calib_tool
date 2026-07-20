# Vision–IMU Calibration — synchronized PC prototype

This tool records a common camera and raw dual-IMU calibration dataset without using the production exoskeleton telemetry path.

```text
USB webcam ───────────────────────────────┐
                                          ├─> PC Vision–IMU tool
UNO R4 + thigh/shank IMUs ── USB Serial ─┘
```

The PC runs MoveNet SinglePose Thunder, draws the live side-view leg overlay, records the visual knee angle, receives raw IMU samples, and aligns both streams using the same PC monotonic clock.

## Isolation from the production system

This milestone does not change:

- production STM32 firmware,
- normal `TELEM` schema,
- serial GUI parser,
- motor or SimpleFOC path,
- safety state flow,
- assistance logic.

The UNO R4 uses a dedicated standalone sketch located at:

```text
standalone_tools/vision_imu_calibration/arduino/
└── uno_r4_dual_imu_stream/
    └── uno_r4_dual_imu_stream.ino
```

## IMU hardware contract

The standalone sketch mirrors the current master firmware sensor configuration:

- thigh IMU: LSM6DSOX at `0x6A`,
- shank IMU: LSM6DSOX at `0x6B`,
- accelerometer: `104 Hz`, `±2 g`,
- gyroscope: `104 Hz`, `±250 dps`,
- acquisition stream: `100 Hz`,
- serial: `921600 baud`.

For UNO R4 WiFi with the Modulino/Qwiic chain, the sketch uses `Wire1`. For sensors connected to the normal SDA/SCL pins, set `IMUCAL_USE_WIRE1` to `0` before compiling.

## Windows setup

From the project root:

```bat
08_SETUP_VISION_IMU_PC.bat
10_TEST_VISION_IMU_PC.bat
```

The isolated `.venv-vision` installs MoveNet/LiteRT, OpenCV, Flask, pytest, and pyserial.

## Fast webcam startup

The default Windows camera profile is intentionally lightweight:

```text
640 × 480 @ 30 FPS
auto backend order: DSHOW → MSMF → OpenCV default
```

MoveNet resizes its input to `256 × 256`, so starting the webcam at 1280 × 720
does not materially improve pose inference but often increases USB camera negotiation time.
The active backend is shown in the runtime source field, for example `camera:0 [DSHOW]`.

For a known camera/backend, startup can be made fully deterministic:

```bat
.venv-vision\Scripts\python.exe -m computer.vision_imu_calibration --camera 0 --camera-backend dshow
```

Fallback values are `msmf`, `default`, and `auto`.

## Vision-only run

```bat
09_RUN_VISION_IMU_PC.bat
```

## Synchronized camera + IMU run

1. Upload this exact sketch to the UNO R4:

```text
standalone_tools/vision_imu_calibration/arduino/
uno_r4_dual_imu_stream/uno_r4_dual_imu_stream.ino
```

2. Close Arduino Serial Monitor.
3. Connect the webcam to the PC.
4. Run:

```bat
11_RUN_VISION_IMU_SYNC_PC.bat
```

5. Enter the UNO R4 COM port, for example `COM3`.
6. Open `http://127.0.0.1:8765`.
7. Wait until the UI shows:
   - pose backend `RUNNING`,
   - IMU serial `CONNECTED`,
   - thigh and shank `READY`.
8. Select the camera-near leg explicitly.
9. Press **Start recording**, remain still for five seconds, then perform five slow flexion–extension cycles.
10. Press **Stop**.

## Output contract

Each session creates:

```text
vision_calibration_recordings/vision_imu_session_<timestamp>/
├── annotated_pose.mp4          # or AVI fallback
├── pose_timeline.csv           # visual angle, landmarks, PC timestamp
├── raw_imu.csv                 # raw + converted IMU values, MCU and PC timestamps
├── synchronized_timeline.csv   # each camera frame paired to nearest IMU sample
└── recording_metadata.json
```

`host_monotonic_ns` is generated from `time.perf_counter_ns()` for both streams. The MCU timestamp is also retained so later stages can evaluate transport delay and reconstruct the native 100 Hz IMU timeline.

## Calibration protocol

The standalone sketch remains idle until the PC sends:

```text
CMD,<id>,START_CAPTURE
```

It replies with `ACK` and sends fixed-shape `IMUCAL` frames. On stop:

```text
CMD,<id>,STOP_CAPTURE
```

The detailed schema is documented in:

```text
standalone_tools/vision_imu_calibration/imu_calibration_protocol.md
```

## Live tuning

The web panel controls only pose validation and display filtering around the fixed pretrained model:

- fixed tracked leg,
- frame rotation,
- landmark confidence threshold,
- landmark smoothing,
- angle smoothing,
- minimum visible segment,
- maximum accepted angle jump.

These values do not modify the neural-network weights and do not yet tune the exoskeleton IMU filter.

## Acceptance gate for this milestone

A valid hardware trial should show:

- IMU rate close to `100 Hz`,
- both IMUs ready,
- no parser errors,
- no queue drops,
- no unexplained sample-index gaps,
- valid pose tracking above `90%` during the controlled movement,
- synchronized camera rows with low and stable `sync_delta_ms`.

The next milestone after this gate is the Python mirror of `knee_angle.cpp` and offline parameter scoring.
