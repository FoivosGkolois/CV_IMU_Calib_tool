from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .camera_source import OpenCvCameraSource
from .imu_filter import (
    KneeAngleFilterMirror,
    KneeAngleFilterParameters,
    resolve_filter_config_path,
)
from .imu_bridge import BridgeImuSource
from .imu_serial import IMUCAL_BAUD_RATE, SerialImuSource
from .marker_reference import AprilTagReference
from .runtime import VisionRuntime
from .web_app import create_app


def resolve_project_root() -> Path:
    """Return the portable CV_IMU_Calib_tool root from the packaged source location."""
    return Path(__file__).resolve().parents[2]


def _parse_camera_selector(value: str) -> int | str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        index = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("camera must be 'auto' or a non-negative integer") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("camera index must be non-negative")
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision–IMU AprilTag reference calibration tool")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--camera",
        type=_parse_camera_selector,
        default="auto",
        help="OpenCV camera index or auto-detect (default: auto)",
    )
    source.add_argument("--video", type=Path, help="Replay a video file instead of using a webcam")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "dshow", "msmf", "v4l2", "default"),
        default="auto",
        help="Camera backend. Auto selects DSHOW/MSMF on Windows and V4L2 on Linux.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--imu-transport",
        choices=("auto", "serial", "uno-q-bridge", "none"),
        default="auto",
        help=(
            "IMU source. auto preserves the PC workflow: serial when --imu-port is set, "
            "otherwise manual GUI connection. uno-q-bridge is for UNO Q Linux."
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
            "snapshot is used in a standalone tool folder."
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

    camera = OpenCvCameraSource(
        camera_index=None if args.video is not None else args.camera,
        video_path=args.video,
        width=args.width,
        height=args.height,
        fps=args.fps,
        # Fiducial detection must use the true, unmirrored camera geometry.
        mirror=False,
        backend=args.camera_backend,
    )
    reference = AprilTagReference()

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
    project_root = resolve_project_root()
    recording_root = project_root / "video_recordings"
    calibration_root = project_root / "calibration_suggestions"

    runtime = VisionRuntime(
        camera,
        reference,
        recording_root=recording_root,
        calibration_root=calibration_root,
        imu_source=imu_source,
        imu_filter=imu_filter,
        imu_filter_config_path=filter_config_path,
    )
    runtime.start()
    app = create_app(runtime)
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}"
    print(f"Vision–IMU tag calibration tool: {url}")
    print(f"Reference backend: {reference.description}")
    if imu_transport == "uno-q-bridge":
        imu_description = "UNO Q Arduino Bridge"
    elif imu_transport == "serial":
        imu_description = f"USB serial {args.imu_port}"
    else:
        imu_description = "manual GUI serial connection"
    print(f"IMU source: {imu_description}")
    print(f"IMU filter config: {filter_config_path}")
    print(f"Video recordings: {recording_root}")
    print(f"Calibration suggestions: {calibration_root}")
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
