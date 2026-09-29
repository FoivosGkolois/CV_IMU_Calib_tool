# Vision IMU Calibration — UNO Q App Lab

This App runs the final thesis calibration tool entirely on the UNO Q: the MCU acquires the two IMUs and the Linux side handles camera/tags, calibration, replay, optimization, recording and reports. The PC is only a browser.

Build from the project root with:

```text
13_BUILD_UNO_Q_PHASE2_APP.bat
```

Then import `uno_q_phase2_app_ready.zip` in Arduino App Lab and press **Run**.

## Browser GUI

Wait for:

```text
[VISION-IMU] WEB GUI READY
```

Then open:

```text
http://<UNO-Q-IP>:8765
```

Use the UNO Q IP shown by App Lab / your normal SSH connection. A container-internal `172.x.x.x` address is not a browser address for the PC. Port 8765 is exposed by `app.yaml`.

## Persistent one-time reference setup

The App stores the reference setup in the Arduino App reserved persistent data folder:

```text
data/reference_setup/reference_setup.yaml
data/reference_setup/REFERENCE_SETUP_LOG.txt
```

It contains:
- checkerboard camera calibration,
- camera/reference geometry,
- the one-time known-angle Tag 0 / Tag 1 validation result.

Normal Stop/Run and UNO Q power cycles do not require repeating this setup. Repeat it only if the physical camera/lens or its geometry/resolution changes, or if the tag mounting contract changes.

## Routine calibration

After the one-time setup has passed:

1. Static: fully extended and still for 12 s.
2. Dynamic: 4 slow + 4 normal + 4 brisk complete flexion-extension cycles.
3. The tool automatically replays the same raw IMU data, runs Bayesian Optimization and validates the result on held-out cycles.
4. Results are shown in the browser and exported to the calibration run folder.

The exact frozen protocol and rationale are packaged with the Python project in `FINAL_PROTOCOL_AND_RATIONALE.md`.
