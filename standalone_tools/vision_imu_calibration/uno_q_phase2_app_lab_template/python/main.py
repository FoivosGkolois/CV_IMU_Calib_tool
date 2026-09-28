from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path

from arduino.app_utils import App

APP_PYTHON_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_PYTHON_ROOT / "project"
WEB_PORT = 8765
WEB_START_TIMEOUT_S = 30.0

if not (PROJECT_ROOT / "computer" / "vision_imu_calibration" / "__main__.py").is_file():
    raise RuntimeError(
        "Phase 2 project payload is missing. Build the App with "
        "13_BUILD_UNO_Q_PHASE2_APP.bat before importing it into App Lab."
    )

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

_web_failure_lock = threading.Lock()
_web_failure = ""


def _set_web_failure(message: str) -> None:
    global _web_failure
    with _web_failure_lock:
        _web_failure = str(message).strip()


def _get_web_failure() -> str:
    with _web_failure_lock:
        return _web_failure


def _interface_ipv4_addresses() -> list[tuple[str, str]]:
    """Return useful non-loopback IPv4 addresses for copy/paste browser URLs."""
    addresses: list[tuple[str, str]] = []
    try:
        import fcntl
    except ImportError:
        fcntl = None

    if fcntl is not None:
        for _, interface_name in socket.if_nameindex():
            if interface_name.startswith(("lo", "docker", "br-", "veth")):
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    request = struct.pack("256s", interface_name[:15].encode("utf-8"))
                    response = fcntl.ioctl(probe.fileno(), 0x8915, request)
                    address = socket.inet_ntoa(response[20:24])
            except OSError:
                continue
            if address.startswith("127.") or address.startswith("169.254."):
                continue
            addresses.append((interface_name, address))

    if not addresses:
        try:
            infos = socket.getaddrinfo(
                socket.gethostname(),
                None,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        except OSError:
            infos = []
        for info in infos:
            address = str(info[4][0])
            if address.startswith("127.") or address.startswith("169.254."):
                continue
            addresses.append(("network", address))

    unique: dict[str, str] = {}
    for interface_name, address in addresses:
        unique.setdefault(address, interface_name)

    def priority(item: tuple[str, str]) -> tuple[int, str, str]:
        address, interface_name = item
        name = interface_name.lower()
        if "usb" in name or "rndis" in name:
            group = 0
        elif name.startswith(("eth", "en")):
            group = 1
        elif name.startswith(("wlan", "wl")):
            group = 2
        else:
            group = 3
        return group, interface_name, address

    return [(interface_name, address) for address, interface_name in sorted(unique.items(), key=priority)]


def _browser_urls() -> list[tuple[str, str]]:
    return [
        (interface_name, f"http://{address}:{WEB_PORT}")
        for interface_name, address in _interface_ipv4_addresses()
    ]


def _local_web_port_is_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", WEB_PORT), timeout=0.35):
            return True
    except OSError:
        return False


def run_vision_imu() -> None:
    try:
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
                str(WEB_PORT),
                "--no-browser",
            ]
        )
    except BaseException:
        details = traceback.format_exc().strip()
        _set_web_failure(details)
        print("\n[VISION-IMU] WEB RUNTIME FAILED", flush=True)
        print(details, flush=True)


def monitor_web_runtime(server_thread: threading.Thread) -> None:
    urls = _browser_urls()
    print("\n[VISION-IMU] Starting web GUI on port 8765...", flush=True)
    if urls:
        print("[VISION-IMU] Browser URL candidate(s):", flush=True)
        for interface_name, url in urls:
            print(f"  {interface_name}: {url}", flush=True)
    else:
        print(
            "[VISION-IMU] Board IPv4 address not resolved yet; App Lab can show the board IP.",
            flush=True,
        )

    deadline = time.monotonic() + WEB_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if _local_web_port_is_ready():
            current_urls = _browser_urls() or urls
            print("\n[VISION-IMU] WEB GUI READY", flush=True)
            if current_urls:
                print("[VISION-IMU] COPY/PASTE ONE OF THESE LINKS:", flush=True)
                for interface_name, url in current_urls:
                    print(f"  {interface_name}: {url}", flush=True)
            else:
                print(
                    f"[VISION-IMU] Open http://<UNO-Q-IP>:{WEB_PORT}",
                    flush=True,
                )
            print(
                "[VISION-IMU] If the browser says CONNECTION_REFUSED before this READY line, wait.",
                flush=True,
            )
            return

        failure = _get_web_failure()
        if failure or not server_thread.is_alive():
            if not failure:
                failure = "Web runtime thread exited before port 8765 became ready."
            print("\n[VISION-IMU] WEB GUI DID NOT START", flush=True)
            print(failure, flush=True)
            print(
                "[VISION-IMU] App Lab may still look active, but the web GUI is unavailable. "
                "Use the error above for diagnosis.",
                flush=True,
            )
            os._exit(1)
        time.sleep(0.25)

    print("\n[VISION-IMU] WEB GUI STARTUP TIMEOUT", flush=True)
    print(
        f"[VISION-IMU] Nothing is listening on local port {WEB_PORT} after "
        f"{WEB_START_TIMEOUT_S:.0f} s.",
        flush=True,
    )
    print(
        "[VISION-IMU] App Lab may still look active, but the web GUI is unavailable. "
        "Check the App launch log above for the first Python error.",
        flush=True,
    )
    os._exit(1)


server_thread = threading.Thread(
    target=run_vision_imu,
    name="vision-imu-web-runtime",
    daemon=True,
)
server_thread.start()
threading.Thread(
    target=monitor_web_runtime,
    args=(server_thread,),
    name="vision-imu-web-monitor",
    daemon=True,
).start()
App.run()
