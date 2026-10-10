#!/bin/sh
# Dedicated standalone browser workflow; stop the normal Vision–IMU service first
# to avoid concurrent access to the same physical camera.
set -eu
cd "$(dirname "$0")"
exec python3 -m computer.vision_imu_calibration --pose-model lightning --inference-threads 4 --imu-transport none --host 0.0.0.0 --port 8765 --no-browser --pose-tuning-only "$@"
