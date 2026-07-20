# Vision–IMU Calibration — synchronized PC prototype

This tool records a common camera and raw dual-IMU calibration dataset without using the production exoskeleton telemetry path.

```text
USB webcam ───────────────────────────────┐
                                          ├─> PC Vision–IMU tool
UNO R4 + thigh/shank IMUs ── USB Serial ─┘
```

The PC runs MoveNet SinglePose Thunder, draws the live side-view leg overlay, receives raw IMU samples, runs a parity-tested Python mirror of the current STM32 `knee_angle.cpp`, and presents both angle timelines without performing parameter optimization.

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

The isolated `.venv-vision` installs MoveNet/LiteRT, OpenCV, Flask, pytest, pyserial, PyYAML, and Matplotlib.

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

5. Open `http://127.0.0.1:8765`.
6. In **UNO R4 raw IMU stream**, press **Refresh**, select the UNO R4 COM port, and press **Connect**.
7. Wait until the UI shows:
   - pose backend `RUNNING`,
   - IMU serial `CONNECTED`,
   - thigh and shank `READY`,
   - current IMU estimator `VALID`.
8. Select the camera-near leg explicitly.
9. Press **Start recording**, remain still for five seconds, then perform five slow flexion–extension cycles.
10. Press **Stop**.

## Output contract

Each completed session creates exactly one archive:

```text
recordings/vision_imu_session_<timestamp>.zip
```

The ZIP contains these files directly at its root:

```text
annotated_pose.mp4              # or AVI fallback
pose_timeline.csv               # visual landmarks and visual posterior knee angle
raw_imu.csv                     # raw and unit-converted IMU measurements
imu_estimator_timeline.csv      # current IMU filter output at IMU rate
synchronized_timeline.csv       # camera rows paired to raw and processed IMU samples
acquisition_summary.json        # descriptive coverage and synchronization evidence
acquisition_overview.png        # visual and IMU angle presentation
recording_metadata.json
```

A temporary workspace is used only while recording and is removed after successful ZIP finalization. Raw and processed data remain in separate files. `host_monotonic_ns` is generated from `time.perf_counter_ns()` for both streams, while the MCU timestamp is retained for the native 100 Hz IMU timeline.

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

This milestone now includes the Python mirror of `knee_angle.cpp` and live/offline presentation. The next milestone is the separate analysis layer that scores candidate parameters and produces recommendations.

## Manual UNO R4 serial connection

The calibration web page owns the standalone IMU serial connection:

1. Upload `standalone_tools/vision_imu_calibration/arduino/uno_r4_dual_imu_stream/uno_r4_dual_imu_stream.ino`.
2. Close Arduino Serial Monitor and any other program using the COM port.
3. Start the calibration application with `11_RUN_VISION_IMU_SYNC_PC.bat`.
4. In the **UNO R4 raw IMU stream** card, press **Refresh**, select the UNO R4 COM port, and press **Connect**.
5. Wait for `CONNECTED`, `Thigh: READY`, and `Shank: READY` before recording.

The COM connection can be changed only while the recorder is idle. This prevents a trial from mixing data from different serial sessions.

## Camera quality profile

The live page exposes two restartable camera profiles:

- **Fast — 640×480:** faster initialization and lower CPU/USB load.
- **Quality — 1280×720:** more source detail and potentially better tracking at difficult side-view angles.

Stop recording before changing profile. The camera briefly restarts, while the MoveNet backend and the rest of the application remain loaded.


## Current IMU estimator presentation

After **Connect**, the UNO R4 stream runs continuously, even while the recorder is idle. The web page shows:

- current IMU posterior knee angle,
- current knee speed,
- thigh and shank filtered orientations,
- visual and IMU angle on a rolling 15 s plot,
- the active standalone parameter source (`default_knee_filter.yaml`).

The **Reload IMU filter config** button resets only the host-side estimator and reloads the standalone filter parameters. It does not write parameters to the Arduino.

The host implementation is identified as `stm32_knee_angle_cpp_mirror_v1` and is checked against the current C++ implementation when a desktop C++ compiler is available.

## Scope boundary

This stage performs acquisition, deterministic filtering, synchronization, recording, and presentation only. It does not search candidate parameters, calculate an optimization score, or generate a recommendation YAML.

## Standalone calibration-tool copy

The acquisition/presentation tool can run outside the complete thesis repository.
At startup it resolves the knee-angle filter configuration in this order:

1. an explicit `--filter-config <path>` argument,
2. `settings/current.yaml` when the complete thesis tree is present,
3. the bundled `default_knee_filter.yaml` snapshot from `Thesis_master(40)`.

The active path is printed in the console and stored with the acquisition metadata.
The C++ parity test runs only when the STM32 source tree is available. Plot export uses
the non-interactive Matplotlib `Agg` backend and does not require Tk/Tkinter.


## Angle convention

Both the visual estimator and the IMU estimator now use the same posterior-knee-angle convention:

- **180°** = straight leg
- **smaller angle** = more knee flexion

The visual overlay also draws a live colored arc behind the knee.
