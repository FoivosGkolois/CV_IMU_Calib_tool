# UNO Q Phase 2 App Lab template

This App runs the complete Vision–IMU tool on the UNO Q Linux processor and the 100 Hz dual-IMU acquisition on its STM32 MCU.

Do not edit or run this template directly. From the project root, run:

```text
13_BUILD_UNO_Q_PHASE2_APP.bat
```

The builder creates `uno_q_phase2_app_ready.zip` with the current Python package, App files, and MCU sketch. Import/open that generated App in Arduino App Lab.

## Opening the browser GUI

After pressing **Run**, use the **App launch** terminal at the bottom of Arduino App Lab.

Do **not** guess when the web server is ready. Wait until the terminal prints:

```text
[VISION-IMU] WEB GUI READY
[VISION-IMU] COPY/PASTE ONE OF THESE LINKS:
  <interface>: http://<UNO-Q-IP>:8765
```

Copy the URL that matches the UNO Q IPv4 address shown by App Lab and paste it into the PC browser.

If the browser reports `ERR_CONNECTION_REFUSED` **before** the `WEB GUI READY` line appears, the server is not ready yet. If startup fails, the App launch terminal now prints `WEB GUI DID NOT START` or `WEB GUI STARTUP TIMEOUT` together with the relevant Python error instead of leaving the App apparently running without a web service.

The App exposes TCP port `8765` in `app.yaml` and the Vision–IMU server binds to `0.0.0.0:8765`.

The USB camera must be visible to the UNO Q App. The launcher defaults to automatic UVC camera discovery instead of assuming `/dev/video0`. It probes the available `/dev/video*` nodes, rejects codec-only devices, and keeps the first node that returns a real frame. `VISION_IMU_CAMERA_INDEX` can still force a specific index.

Phase 1 and the original PC workflows remain separate and unchanged.

Outputs are stored inside the packaged project root using deterministic owners:

```text
video_recordings/
calibration_suggestions/calibration_run_<timestamp>/
```

The web GUI provides clickable project-relative export links. One calibration session contains `static_trial.zip`, `dynamic_trial.zip`, and one atomically updated `imu_filter_suggestions.yaml`.
