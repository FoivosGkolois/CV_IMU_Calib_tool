# Vision–IMU Calibration — synchronized PC prototype

This tool records a common camera and raw dual-IMU calibration dataset without using the production exoskeleton telemetry path.

```text
USB webcam ───────────────────────────────┐
                                          ├─> PC Vision–IMU tool
UNO R4 + thigh/shank IMUs ── USB Serial ─┘
```

The PC runs MoveNet SinglePose Thunder, draws the live side-view leg overlay, receives raw IMU samples, runs a parity-tested Python mirror of the current STM32 `knee_angle.cpp`, and presents both angle timelines. A guided offline layer can replay the recorded raw IMU data and export comparative recommended filter settings after the static and dynamic tests.

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
The default camera selector is `auto`. The tool tests available camera indices and
keeps the first source that returns a real frame. On Linux it prioritizes UVC/webcam
nodes and de-prioritizes internal encoder/decoder devices. The active index and
backend are shown in the runtime source field, for example `camera:2 [V4L2, MJPG]`.

For a known camera/backend, startup can be made fully deterministic:

```bat
.venv-vision\Scripts\python.exe -m computer.vision_imu_calibration --camera 0 --camera-backend dshow
```

Fallback values are `msmf`, `v4l2`, `default`, and `auto`.

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
video_recordings/vision_imu_session_<timestamp>.zip
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

These values do not modify the neural-network weights. The calibration workflow uses the active pose-reference settings but never changes them automatically during a test.

## Acceptance gate for this milestone

A valid hardware trial should show:

- IMU rate close to `100 Hz`,
- both IMUs ready,
- no parser errors,
- no queue drops,
- no unexplained sample-index gaps,
- valid pose tracking above `90%` during the controlled movement,
- synchronized camera rows with low and stable `sync_delta_ms`.

This milestone includes the Python mirror of `knee_angle.cpp`, live/offline presentation, and the guided static/dynamic recommendation workflow.

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

The live runtime performs acquisition, deterministic filtering, synchronization, recording, and presentation. Parameter search runs only offline after both guided tests finish. It uses a limited deterministic grid, preserves the confirmed axis/sign conventions, and exports a recommendation YAML without writing anything to the Arduino or production firmware.

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


## Responsive live workspace

The camera, metric cards, angle plot, and control cards automatically fit the available browser width. Their size is controlled by the responsive page layout rather than user drag handles, so the same GUI behaves deterministically across PCs and the UNO Q browser client.

The live video overlay includes the compact tracking badge plus the thigh/shank bars, hip-knee-ankle markers, posterior knee-angle arc, and live angle value.

## Calibration and independent held-out validation

For the initial laboratory session, optionally collect the four physical-goniometer points, then run **Position Subject** and mark the subject/camera locations. Enter one setup ID/description and keep the same person, IMU mounting, selected leg, camera placement/profile and pose settings through calibration and held-out validation. Both calibration stages require the IMU stream to be connected, both IMUs to be ready, and visual tracking to be valid.

### Stage 1 — static

Press **Start calibration**, read the instructions, and start the five-second countdown. The tool records automatically for 12 seconds. Keep the complete hip–knee–ankle chain visible and hold a comfortable near-straight angle completely still. The measured optical angle is retained; it is not forced to 180 degrees.

The static stage estimates the common knee-angle offset, thigh and shank gyro biases, and a practical gyro deadband. It writes the first section of the shared `imu_filter_suggestions.yaml` and leaves the session in `AWAITING_DYNAMIC`.

### Stage 2 — dynamic

Press **Continue to dynamic calibration**. After the five-second countdown, perform about five slow and smooth flexion–extension cycles. The tool records automatically for 25 seconds.

The PC freezes the synchronized training pairs and runs 30 deterministic Gaussian-process Bayesian evaluations of only the three dynamic filter parameters. It compares every candidate on the same pairs at fixed host lag 0 ms and completes the same YAML with the dynamic suggestion and final merged settings.

The workflow preserves the confirmed axis mapping and sign conventions. It does not rewrite hardware orientation parameters automatically.

### Stage 3 — new held-out trial

After Bayesian tuning finishes, use the existing **Start recording** and **Stop** controls for a new slow flexion–extension trial. Confirm the setup is unchanged, then press **Validate latest new recording**. The tool replays the frozen original, static-calibrated and BO-recommended snapshots on exactly the same new pairs; it does not refit bias, offset, parameters, visual angles or lag. `IMPROVED_AGREEMENT` means the BO snapshot has lower held-out RMSE than the static-calibrated snapshot. `NO_IMPROVEMENT` is retained honestly when training improved but the new trial did not. The comparison is not clinical accuracy or proof of camera accuracy; fixed host lag 0 ms does not establish zero camera/inference latency.

### Deterministic output ownership

The normal **Record Video** control owns only:

```text
video_recordings/vision_imu_session_<timestamp>.zip
```

Each calibration session owns one run directory:

```text
calibration_suggestions/calibration_run_<timestamp>/
```

The run contains:

```text
static_trial.zip
dynamic_trial.zip
imu_filter_suggestions.yaml
calibration_report.json
calibration_overview.png
heldout_validation.json
```

Both trial archives include their own video, raw IMU, processed IMU, synchronized timeline, metadata, and overview files. Static and dynamic never write to `video_recordings`, and the simple recorder never writes to `calibration_suggestions`.

Paths shown in the GUI are project-relative and clickable through the web server, so no Windows-specific or UNO-Q-specific absolute path is exposed. The YAML is written atomically: after the static stage it contains the static suggestion and `awaiting_dynamic`; after the dynamic stage the same file contains the dynamic suggestion and final merged settings.

If quality gates fail, the shared YAML records the invalid state and the failed gates rather than presenting an unvalidated final configuration.

## UNO Q Phase 2 transport

The runtime now supports two interchangeable IMU transports:

```text
USB serial       --imu-transport serial --imu-port COMx
UNO Q Bridge     --imu-transport uno-q-bridge
```

`BridgeImuSource` preserves the existing `ImuCalibrationSample` contract, so pose processing, filtering, synchronization, recording, guided static/dynamic tests, and exported settings do not depend on the transport.

For UNO Q deployment, run `13_BUILD_UNO_Q_PHASE2_APP.bat` from the project root and open the generated App in Arduino App Lab.
# Optional subject positioning and goniometer comparison (Ticket 1)

Open **Optional vision setup / laboratory validation** in the dashboard.
**Position Subject** adds a fixed normalized hip cross and 5% frame-edge safe
area to the existing, already-rotated camera stream. Mark the foot/chair location
on the floor and record camera mounting/distance. The operator must manually
confirm side-on orientation; 2-D landmarks cannot establish anatomical side-on.
Start the slow sweep, flex to approximately 90 degrees, extend toward 175, return,
then Finish. The check needs at least 3 seconds and 15 valid observations, gaps
no greater than 0.75 seconds, hip within 10% of the target, and alternating
flexed (<=110), extended (>=165), then returned endpoints. These tolerances
allow approximate positioning; Passed applies only to the observed movement.
A missing/low-confidence/off-frame observation requires Retry. Passed retains
thin fixed guides; a hip shift greater than 6% or visibility loss warns without
cancelling ordinary recordings. Camera profile, rotation, tuning, actual tracked
leg changes and tracking restart reset Passed.

For independent static comparison, describe subject ID, tripod/mounting/distance,
floor mark/chair and goniometer. Choose a **prompt** (90/120/150/175), select
**New / repeat point**, set the limb with a physical goniometer, and enter its
actual measured posterior angle (180 = straight). The live image keeps its
skeleton/arc but hides numerical AI angles; summary/chart data are blinded too.
Confirm the manual value to reveal the AI readings, then Capture a 3-second
stationary hold. Raw and displayed angles are preserved without correction.
The existing recorder writes each hold to a unique ZIP under
`video_recordings/goniometer/`; the adjacent JSON contains measured inputs,
profile/setup, raw landmark observations, timestamps, SHA-256, signed AI-minus-
measured errors, frame coverage and per-prompt repeatability for the same setup.
It reuses `reference_validation.py` statistics and preserves ordinary ZIP/CSV schemas.
Hold acceptance needs >=15 valid frames spanning >=2 seconds, no invalid frames
or gaps >0.75 seconds (including hold edges), raw/displayed angle range <=3 degrees
and raw landmark coordinate range <=2% of the frame. These conservative lab
heuristics reject motion/dropouts; **Measured** does not certify accuracy.
Failed attempts keep their own artifacts and do not replace accepted captures.
Repeat a point freely. Stop the ordinary recording/calibration before a hold.

Saved accepted reports reload on restart. Select a saved capture, reproduce its
described setup in the setup field, and explicitly confirm unchanged physical
setup. Reuse requires matching camera/pose/tuning/tracked-leg metadata and an
unchanged recording hash. Routine trials do not require a goniometer or a Passed
positioning check. Keep both optional features provisional until UNO Q checks:
rotated overlay alignment, profile/leg/restart resets, live FPS, four real
goniometer holds, failed repeat, export and restart/reuse.
