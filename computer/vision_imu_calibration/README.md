# Vision–IMU Calibration — V2 marker-reference migration

This tool is migrating from learned human-pose estimation to a fixed two-tag geometric reference.

Current measurement contract:

```text
USB camera -> AprilTag detector -> calibrated 3-D tag poses
Tag 0 -> thigh
Tag 1 -> shank
relative 3-D segment orientation -> posterior knee angle
```

The previous MoveNet model, landmark smoothing, confidence threshold, leg selection, and angle-smoothing controls are removed from the active calibration path.

## Current reference-setup state

Patch 1.2 provides the calibrated 3-D two-tag geometry and fixed reference-quality gates.
Patch 2.1 provides the checkerboard camera-calibration workflow. Patch 2.2 persists a
successful setup in `reference_setup.yaml`, reloads it on startup, validates the saved
camera geometry/resolution before use, and records setup history in
`REFERENCE_SETUP_LOG.txt`. No uncalibrated 2-D fallback is allowed.

## Mounting contract

- Dictionary: AprilTag 36h11
- Tag 0: THIGH
- Tag 1: SHANK
- black-square side: 70 mm
- tags rigidly mounted to their carriers
- the image-top edge of each tag points proximally along its segment
  - thigh -> toward hip
  - shank -> toward knee

The relative angle uses the 3-D proximal axes of the two tags, so common body rotation relative to the camera largely cancels.

## Migration safety gate

Patch 3 enables the scientific calibration workflow. The static stage derives straight-leg offset, gyro bias/noise and deadband from the known full-extension condition and stationary IMU physics; the marker reference is only an optional sanity check in this stage. The dynamic stage detects complete marker-reference cycles, selects temporary grid-search candidates only on training cycles, and reports final performance on held-out cycles that were not used for parameter selection. Candidate-specific time shifting is not used in the primary RMSE objective.

Normal recording remains available throughout the migration.

## What remains unchanged

- camera auto-detection and camera profiles,
- UNO Q Bridge / serial IMU acquisition,
- raw IMU schema and 100 Hz acquisition contract,
- host mirror of the STM32 knee-angle filter,
- recorder ownership and export roots,
- static -> dynamic calibration session ownership,
- synchronized timelines and calibration run folders.

## Dependencies

PC:

```text
opencv-contrib-python
```

UNO Q App Lab:

```text
opencv-contrib-python-headless
```

OpenCV contrib is required for the AprilTag dictionary support.

## Required printed reference targets

Printable AprilTag assets and the mounting contract are under:

```text
reference_setup/tags/
```

The Patch 2.1 checkerboard and exact-size printing instructions are under:

```text
reference_setup/checkerboard/
```

## Normal outputs

Simple recorder:

```text
video_recordings/
```

Calibration run ownership remains:

```text
calibration_suggestions/calibration_run_<timestamp>/
```

The reference timeline is:

```text
reference_timeline.csv
```

and the annotated video is:

```text
annotated_reference.mp4
```

(or AVI fallback).

## Next patch

Patch 4 replaces the temporary deterministic grid search with Gaussian-process
Bayesian Optimization while preserving the deterministic replay, Patch-3 cycle split,
and held-out evaluation contract. Patch 4.2 adds final uncertainty/confidence
reporting and the final browser/YAML presentation.
