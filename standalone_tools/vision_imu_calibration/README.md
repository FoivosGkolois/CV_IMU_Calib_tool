# Standalone Vision–IMU calibration hardware

This folder contains the temporary UNO R4 acquisition firmware used to validate camera-assisted IMU calibration before the UNO Q Bridge/RPC integration.

Upload:

```text
arduino/uno_r4_dual_imu_stream/uno_r4_dual_imu_stream.ino
```

The sketch initializes only the two LSM6DSOX IMUs and the USB serial link. It contains no motor, encoder, EMG, FSR, assistance, or safety-control code.

Use it with:

```text
11_RUN_VISION_IMU_SYNC_PC.bat
```

The standalone acquisition path is intentionally separate from the production firmware so it cannot modify the established telemetry or control lifecycle.
