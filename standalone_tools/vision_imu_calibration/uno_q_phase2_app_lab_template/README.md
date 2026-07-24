# UNO Q Phase 2 App Lab template

This App runs the complete Vision–IMU tool on the UNO Q Linux processor and the 100 Hz dual-IMU acquisition on its STM32 MCU.

Do not edit or run this template directly. From the project root, run:

```text
13_BUILD_UNO_Q_PHASE2_APP.bat
```

The builder creates `uno_q_phase2_app_ready.zip` with the current Python package, model, App files, and MCU sketch. Import/open that generated App in Arduino App Lab.

After the App starts, open in a PC browser:

```text
http://<UNO-Q-IP>:8765
```

The USB camera must be visible to the UNO Q App. The launcher now defaults to
automatic UVC camera discovery instead of assuming `/dev/video0`. It probes the
available `/dev/video*` nodes, rejects codec-only devices, and keeps the first node
that returns a real frame. `VISION_IMU_CAMERA_INDEX` can still force a specific index.

Phase 1 and the original PC workflows remain separate and unchanged.
