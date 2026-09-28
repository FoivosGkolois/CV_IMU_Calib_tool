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

## Patch 1.2 state

Patch 1.2 adds the deterministic 3-D geometry layer and fixed reference-quality gates:

- calibrated square-tag pose with `SOLVEPNP_IPPE_SQUARE`,
- Tag 0 / Tag 1 3-D poses,
- relative thigh/shank segment angle,
- posterior knee convention: 180° at collinear extension,
- camera-resolution consistency check,
- minimum observed marker size,
- maximum viewing obliquity,
- normalized reprojection-error check,
- per-frame pose-quality metrics in the reference recording.

The angle is intentionally **invalid until calibrated camera intrinsics are supplied**. Patch 2.1 will add the one-time checkerboard workflow and persistent `reference_setup.yaml` that loads those intrinsics automatically.

This means Patch 1.2 can be unit-tested completely, while normal hardware operation still reports `reference_setup_required` until Patch 2.1 is applied. No uncalibrated 2-D fallback is allowed.

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

The legacy static/dynamic calibration analysis still exists only to preserve the current branch while migration proceeds. It remains explicitly disabled. Patch 3 will replace that analysis with the physics-based static stage and the new dynamic validation logic before scientific IMU tuning is re-enabled.

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

## Required printed markers

Printable tag assets and the mounting contract are under:

```text
reference_setup/tags/
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

Patch 2.1 adds:

1. checkerboard camera calibration,
2. persistent `reference_setup.yaml`,
3. automatic loading and validation of the saved camera profile,
4. activation of the Patch 1.2 3-D reference on matching camera/resolution.
