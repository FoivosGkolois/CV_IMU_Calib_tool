from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from arduino.app_utils import App

APP_PYTHON_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_PYTHON_ROOT / "project"

if not (PROJECT_ROOT / "computer" / "vision_imu_calibration" / "__main__.py").is_file():
    raise RuntimeError(
        "Phase 2 project payload is missing. Build the App with "
        "13_BUILD_UNO_Q_PHASE2_APP.bat before importing it into App Lab."
    )

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def run_vision_imu() -> None:
    from computer.vision_imu_calibration.__main__ import main

    main(
        [
            "--camera",
            os.environ.get("VISION_IMU_CAMERA_INDEX", "auto"),
            "--camera-backend",
            os.environ.get("VISION_IMU_CAMERA_BACKEND", "auto"),
            "--width",
            os.environ.get("VISION_IMU_CAMERA_WIDTH", "640"),
            "--height",
            os.environ.get("VISION_IMU_CAMERA_HEIGHT", "480"),
            "--fps",
            os.environ.get("VISION_IMU_CAMERA_FPS", "30"),
            "--imu-transport",
            "uno-q-bridge",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
            "--no-browser",
        ]
    )


server_thread = threading.Thread(
    target=run_vision_imu,
    name="vision-imu-web-runtime",
    daemon=True,
)
server_thread.start()
App.run()
