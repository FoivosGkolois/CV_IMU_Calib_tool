from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .camera_source import OpenCvCameraSource
from .download_model import DEFAULT_TARGET, download_model
from .imu_filter import (
    KneeAngleFilterMirror,
    KneeAngleFilterParameters,
    resolve_filter_config_path,
)
from .imu_bridge import BridgeImuSource
from .imu_serial import IMUCAL_BAUD_RATE, SerialImuSource
from .models import LegSide
from .pose_estimator import MoveNetThunderPoseEstimator
from .runtime import VisionRuntime
from .web_app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PC-first Vision–IMU pose visualizer")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    source.add_argument("--video", type=Path, help="Replay a video file instead of using a webcam")
    parser.add_argument("--leg", choices=[side.value for side in LegSide], default=LegSide.RIGHT.value)
    parser.add_argument("--model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--inference-threads", type=int, default=4)
    parser.add_argument(
        "--no-smart-crop",
        action="store_true",
        help="Disable MoveNet video crop tracking and process the full padded frame every time",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "dshow", "msmf", "default"),
        default="auto",
        help="Windows camera backend. Auto prefers DSHOW for faster startup.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--recording-root", type=Path, default=Path("recordings"))
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("calibration_results"),
        help="Folder for guided static/dynamic calibration runs and recommended settings",
    )
    parser.add_argument(
        "--imu-transport",
        choices=("auto", "serial", "uno-q-bridge", "none"),
        default="auto",
        help=(
            "IMU source. auto preserves the PC workflow: serial when --imu-port is set, "
            "otherwise manual GUI connection. uno-q-bridge is for Phase 2 on UNO Q Linux."
        ),
    )
    parser.add_argument("--imu-port", help="Calibration serial port, e.g. COM3")
    parser.add_argument("--imu-baud", type=int, default=IMUCAL_BAUD_RATE)
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=None,
        help=(
            "Optional runtime YAML containing stm32_control.knee_angle parameters. "
            "Without this option, settings/current.yaml is preferred and the bundled "
            "Thesis_master(40) snapshot is used in a standalone tool folder."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if not (sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 13):
        raise SystemExit(
            "This prototype requires Python 3.10-3.13. "
            f"Current interpreter: {sys.version.split()[0]}"
        )
    args = build_parser().parse_args(argv)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        download_model(model_path)

    camera = OpenCvCameraSource(
        camera_index=None if args.video is not None else args.camera,
        video_path=args.video,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=not args.no_mirror,
        backend=args.camera_backend,
    )
    estimator = MoveNetThunderPoseEstimator(
        model_path,
        num_threads=args.inference_threads,
        smart_crop=not args.no_smart_crop,
    )
    imu_transport = args.imu_transport
    if imu_transport == "auto":
        imu_transport = "serial" if args.imu_port else "none"
    if imu_transport == "serial":
        if not args.imu_port:
            raise SystemExit("--imu-transport serial requires --imu-port")
        imu_source = SerialImuSource(args.imu_port, baud_rate=args.imu_baud)
    elif imu_transport == "uno-q-bridge":
        if args.imu_port:
            raise SystemExit("--imu-port cannot be combined with --imu-transport uno-q-bridge")
        imu_source = BridgeImuSource()
    else:
        imu_source = None
    filter_config_path = resolve_filter_config_path(args.filter_config)
    filter_parameters = KneeAngleFilterParameters.load_yaml(filter_config_path)
    imu_filter = KneeAngleFilterMirror(filter_parameters)
    runtime = VisionRuntime(
        camera,
        estimator,
        leg_side=LegSide(args.leg),
        confidence_threshold=args.confidence,
        recording_root=args.recording_root,
        calibration_root=args.calibration_root,
        imu_source=imu_source,
        imu_filter=imu_filter,
        imu_filter_config_path=filter_config_path,
    )
    runtime.start()
    app = create_app(runtime)
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}"
    print(f"Vision–IMU calibration prototype: {url}")
    print(f"Pose backend: {estimator.description}")
    if imu_transport == "uno-q-bridge":
        imu_description = "UNO Q Arduino Bridge"
    elif imu_transport == "serial":
        imu_description = f"USB serial {args.imu_port}"
    else:
        imu_description = "manual GUI serial connection"
    print(f"IMU source: {imu_description}")
    print(f"IMU filter config: {filter_config_path}")
    print("Stop with Ctrl+C.")
    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        runtime.stop()
    return 0


def _open_browser(url: str) -> None:
    time.sleep(1.0)
    webbrowser.open(url, new=2)


if __name__ == "__main__":
    raise SystemExit(main())
