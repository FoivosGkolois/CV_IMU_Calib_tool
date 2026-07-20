from __future__ import annotations

import pytest

pytest.importorskip("flask")

from computer.vision_imu_calibration.web_app import create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.connected: tuple[str, int] | None = None
        self.profile = "fast"

    def status(self):
        return {"state": "RUNNING"}

    def tuning(self):
        return {}

    def update_tuning(self, payload):
        return payload

    def list_imu_ports(self):
        return [{"device": "COM4", "description": "UNO R4", "hwid": "USB"}]

    def connect_imu(self, port: str, *, baud_rate: int):
        self.connected = (port, baud_rate)
        return {"configured": True, "port": port, "baud_rate": baud_rate}

    def disconnect_imu(self):
        self.connected = None
        return {"configured": False, "state": "DISCONNECTED"}

    def update_camera_profile(self, profile: str):
        self.profile = profile
        return {"profile": profile, "width": 1280, "height": 720, "fps": 30.0}

    def start_recording(self):
        return {"active": True}

    def stop_recording(self):
        return {"active": False}

    def wait_for_jpeg(self, after_sequence: int, timeout_s: float = 1.0):
        return after_sequence, None


def test_manual_imu_and_camera_profile_endpoints() -> None:
    runtime = FakeRuntime()
    app = create_app(runtime)  # type: ignore[arg-type]
    client = app.test_client()

    ports = client.get("/api/imu/ports")
    assert ports.status_code == 200
    assert ports.get_json()["ports"][0]["device"] == "COM4"

    connected = client.post(
        "/api/imu/connect",
        json={"port": "COM4", "baud_rate": 921600},
    )
    assert connected.status_code == 200
    assert runtime.connected == ("COM4", 921600)

    camera = client.post("/api/camera/profile", json={"profile": "quality"})
    assert camera.status_code == 200
    assert runtime.profile == "quality"

    disconnected = client.post("/api/imu/disconnect")
    assert disconnected.status_code == 200
    assert runtime.connected is None
