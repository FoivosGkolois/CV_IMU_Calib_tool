# Vision–IMU Calibration — V2 tag-reference transition

This tool is migrating from learned human-pose estimation to a fixed two-tag geometric reference.

Current V2.1 measurement contract:

```text
USB camera -> AprilTag detector
Tag 0 -> thigh
Tag 1 -> shank
```

The previous MoveNet model, landmark smoothing, confidence threshold, leg selection, and angle-smoothing controls are removed from the active calibration path.

## Important V2.1 gate

V2.1 detects the two required tags and records their detections, but it intentionally does **not** produce a scientific knee-angle reference yet. The knee-angle output stays invalid until the persistent calibrated 3-D reference setup is added in the next patch.

This prevents an uncalibrated 2-D angle from silently becoming the new reference.

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

OpenCV contrib is required because the reference detector uses the built-in AprilTag dictionary support.

## Required printed markers

- Tag 0: THIGH
- Tag 1: SHANK
- Dictionary: AprilTag 36h11

Printable tag assets are included under:

```text
reference_setup/tags/
```

## Normal outputs

Simple recorder:

```text
video_recordings/
```

Calibration run:

```text
calibration_suggestions/calibration_run_<timestamp>/
```

The visual/reference timeline inside each recording is now named:

```text
reference_timeline.csv
```

and the annotated video is:

```text
annotated_reference.mp4
```

(or AVI fallback).

## Next V2 patch

The next patch adds:

1. one-time checkerboard camera calibration,
2. persistent `reference_setup.yaml`,
3. 3-D thigh/shank tag pose,
4. relative knee-angle calculation,
5. reference quality gates and persistence across normal UNO Q reboots.

Only after that reference is valid should the scientific calibration workflow be enabled again.
