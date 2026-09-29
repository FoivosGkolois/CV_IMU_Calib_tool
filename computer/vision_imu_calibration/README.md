# Vision–IMU Calibration Tool — final thesis architecture

This is the standalone calibration and validation environment for the dual-IMU knee-angle estimator used in the knee-assist exoskeleton thesis.

## What the tool does

1. A USB camera observes two rigid AprilTags: Tag 0 on the thigh and Tag 1 on the shank.
2. One-time checkerboard calibration gives the camera its geometric calibration.
3. One-time known-angle fixture validation measures how accurately and repeatably the complete camera + tag reference reports knee angle.
4. A 12 s stationary full-extension stage measures IMU bias/noise and straight-leg offset from physics.
5. A dynamic trial records 12 complete flexion-extension cycles at slow, normal and brisk speeds.
6. The exact same raw IMU recording is replayed through many filter settings.
7. Gaussian-Process Bayesian Optimization selects filter settings using only training cycles.
8. Three cycles, one from each speed block, remain unseen until final validation.
9. The browser, YAML, JSON and overview plot report the recommendation and validation evidence.

The full frozen protocol and its rationale are in `FINAL_PROTOCOL_AND_RATIONALE.md`.

## Execution split

```text
UNO Q MCU   -> deterministic dual-LSM6DSOX acquisition at ~100 Hz
Arduino Bridge (921600 baud) -> raw samples to UNO Q Linux
UNO Q Linux -> camera/tags, calibration, replay, optimization, logging, reports
PC          -> browser only
```

No motor control is included in this App Lab calibration application.

## Reference contract

- Dictionary: AprilTag 36h11
- Tag 0: THIGH
- Tag 1: SHANK
- black-square side: 70 mm
- rigid, flat mounting
- tag top edge points proximally along the segment
- posterior knee angle: 180 deg = full extension

There is no uncalibrated 2-D fallback. A valid 3-D reference requires the saved camera calibration.

## Persistent reference setup

The UNO Q stores the one-time setup in its persistent App Lab data area:

```text
reference_setup.yaml
REFERENCE_SETUP_LOG.txt
```

The setup contains camera intrinsics, calibration quality, tag geometry and the known-angle reference-validation result. Ordinary UNO Q power cycles do not require repeating the setup. Recalibration is required if the camera/lens/resolution/rotation geometry changes.

## Routine calibration outputs

```text
calibration_suggestions/calibration_run_<timestamp>/
    static_trial.zip
    dynamic_trial.zip
    reference_setup_snapshot.yaml
    RUN_PROVENANCE.json
    imu_filter_suggestions.yaml
    calibration_report.json
    calibration_overview.png
```

The run stores a snapshot/hash of the active reference setup and build/filter provenance so the final result can be traced back to the exact setup used.

## Scientific boundaries

The marker system is a validated engineering reference, not absolute ground truth. The Bayesian optimizer searches filter parameters; it does not identify gait intent or control the exoskeleton. The tool is an engineering/thesis calibration environment, not a clinical or medical device.
