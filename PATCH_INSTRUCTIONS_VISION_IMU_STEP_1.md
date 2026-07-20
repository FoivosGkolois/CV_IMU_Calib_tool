# Vision–IMU Step 1 patch instructions

Baseline: `Thesis_master(29).zip`, branch `master_2`, commit `7e0806d`.

This patch adds only the isolated PC vision feasibility prototype. It does not modify firmware, serial communication, telemetry, motor, safety, logger, replay, or the existing GUI.

## Apply

Copy every file from this ZIP into the root of the current `Thesis_master` working tree, preserving the included folder structure.

No files are deleted or replaced except `.gitignore`, which receives three new ignore rules for the isolated environment and downloaded model.

## Verify

```bat
08_SETUP_VISION_IMU_PC.bat
10_TEST_VISION_IMU_PC.bat
09_RUN_VISION_IMU_PC.bat
```

## Commit title

```text
Add PC Vision-IMU pose feasibility prototype
```
