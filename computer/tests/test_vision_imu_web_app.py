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

    def open_recordings_location(self):
        return {"path": "C:/tool/recordings"}

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

    opened = client.post("/api/recordings/open")
    assert opened.status_code == 200
    assert opened.get_json()["path"].endswith("recordings")


def test_filter_reload_and_history_endpoints() -> None:
    class ExtendedRuntime(FakeRuntime):
        def reload_imu_filter(self):
            return {"valid": False, "reason": "filter_reloaded"}

        def acquisition_history(self, *, seconds: float, max_points: int):
            return [{"host_monotonic_ns": 1, "visual_valid": True, "imu_valid": True}]

    runtime = ExtendedRuntime()
    app = create_app(runtime)  # type: ignore[arg-type]
    client = app.test_client()
    reload_response = client.post("/api/imu-filter/reload")
    assert reload_response.status_code == 200
    assert reload_response.get_json()["reason"] == "filter_reloaded"
    history = client.get("/api/acquisition/history?seconds=10&max_points=50")
    assert history.status_code == 200
    assert history.get_json()["rows"][0]["host_monotonic_ns"] == 1


def test_page_uses_adaptive_and_user_resizable_video_shell() -> None:
    runtime = FakeRuntime()
    app = create_app(runtime)  # type: ignore[arg-type]
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'id="videoShell"' in html
    assert 'function resizeVideoShell()' in html
    assert "videoStream.naturalWidth" in html
    assert "setInterval(resizeVideoShell,1000)" in html
    assert 'alt="Live camera stream"' in html
    assert 'data-resize-id="camera"' in html
    assert "visionImuCustomBoxSizesV1" in html
    assert "function resetCustomLayout()" in html


def test_page_uses_left_camera_live_dashboard_workspace_without_header() -> None:
    runtime = FakeRuntime()
    app = create_app(runtime)  # type: ignore[arg-type]
    client = app.test_client()

    html = client.get("/").get_data(as_text=True)
    assert "VISION–IMU ACQUISITION & PRESENTATION" not in html
    assert '<section class="camera-column">' in html
    assert '<section class="live-column">' in html
    assert html.index('class="camera-column"') < html.index('class="live-column"')
    assert html.index('class="live-column"') < html.index('class="control-panel"')
    assert "grid-template-columns:minmax(360px,max-content) minmax(360px,1fr) minmax(300px,340px)" in html
    assert "justify-self:start" in html


def test_page_makes_live_plot_and_all_main_cards_user_resizable() -> None:
    runtime = FakeRuntime()
    app = create_app(runtime)  # type: ignore[arg-type]
    client = app.test_client()

    html = client.get("/").get_data(as_text=True)
    expected_ids = (
        "tracking-card", "visual-card", "imu-card", "pose-card", "angle-plot",
        "recording-card", "imu-stream-card", "camera-profile-card",
        "estimator-card", "tuning-card", "diagnostics-card",
    )
    for resize_id in expected_ids:
        assert f'data-resize-id="{resize_id}"' in html
    assert ".resizable-box { resize:both" in html
    assert ".chart-card { padding:12px; height:270px" in html
    assert "Reset all box sizes" in html
