from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

from computer.vision_imu_calibration.web_app import create_app


class FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.recording_root = root / "video_recordings"
        self.calibration_root = root / "calibration_suggestions"
        self.recording_root.mkdir(parents=True)
        self.calibration_root.mkdir(parents=True)
        self.connected: tuple[str, int] | None = None
        self.profile = "fast"
        self.calibration_test = ""

    def status(self):
        return {
            "state": "RUNNING",
            "recording": {
                "active": False,
                "display_path": "video_recordings",
                "download_url": "",
                "browse_url": "/exports/recordings/",
                "error": "",
            },
            "calibration": {
                "active": False,
                "state": "IDLE",
                "current_test": "",
                "message": "Start calibration with the static stage.",
                "error": "",
                "display_path": "calibration_suggestions",
                "browse_url": "/exports/calibration/",
                "recommendation_display_path": "",
                "recommendation_url": "",
                "recommended_parameters": {},
                "static_suggestion": {},
                "dynamic_suggestion": {},
            },
            "imu": {"transport": "serial"},
            "camera_calibration": {
                "active": False,
                "state": "IDLE",
                "message": "Ready",
                "accepted_views": 0,
                "required_views": 15,
                "checkerboard_detected": False,
                "rms_reprojection_error_px": None,
                "persistent": False,
            },
        }

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

    def start_camera_calibration(self):
        return {"active": True, "state": "CAPTURING", "accepted_views": 0, "required_views": 15}

    def cancel_camera_calibration(self):
        return {"active": False, "state": "IDLE"}

    def start_recording(self):
        return {"active": True}

    def stop_recording(self):
        return {"active": False}

    def open_recordings_location(self):
        return {"path": "video_recordings", "url": "/exports/recordings/"}

    def start_calibration_test(self, test_type: str):
        self.calibration_test = test_type
        return {"active": True, "current_test": test_type, "duration_s": 12.0}

    def stop_calibration_test(self):
        self.calibration_test = ""
        return {"active": False, "state": "AWAITING_DYNAMIC"}

    def cancel_calibration_test(self):
        self.calibration_test = ""
        return {"active": False, "state": "IDLE"}

    def open_calibration_location(self):
        return {
            "path": "calibration_suggestions/calibration_run_test",
            "url": "/exports/calibration/calibration_run_test/",
        }

    def wait_for_jpeg(self, after_sequence: int, timeout_s: float = 1.0):
        return after_sequence, None


def test_manual_imu_camera_and_portable_export_endpoints(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
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
    assert opened.get_json() == {
        "path": "video_recordings",
        "url": "/exports/recordings/",
    }


def test_filter_reload_and_history_endpoints(tmp_path: Path) -> None:
    class ExtendedRuntime(FakeRuntime):
        def reload_imu_filter(self):
            return {"valid": False, "reason": "filter_reloaded"}

        def acquisition_history(self, *, seconds: float, max_points: int):
            return [{"host_monotonic_ns": 1, "visual_valid": True, "imu_valid": True}]

    runtime = ExtendedRuntime(tmp_path)
    app = create_app(runtime)  # type: ignore[arg-type]
    client = app.test_client()
    reload_response = client.post("/api/imu-filter/reload")
    assert reload_response.status_code == 200
    assert reload_response.get_json()["reason"] == "filter_reloaded"
    history = client.get("/api/acquisition/history?seconds=10&max_points=50")
    assert history.status_code == 200
    assert history.get_json()["rows"][0]["host_monotonic_ns"] == 1


def test_page_uses_responsive_non_user_resizable_layout(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    html = create_app(runtime).test_client().get("/").get_data(as_text=True)  # type: ignore[arg-type]

    assert 'id="videoShell"' in html
    assert 'alt="Live camera stream"' in html
    assert "ResizeObserver" in html
    assert "aspect-ratio" in html
    assert "resize:both" not in html.replace(" ", "")
    assert "data-resize-id" not in html
    assert "visionImuCustomBoxSizesV1" not in html
    assert "Reset all box sizes" not in html
    assert "Custom layout" not in html


def test_page_uses_fluid_three_column_dashboard_with_breakpoints(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    html = create_app(runtime).test_client().get("/").get_data(as_text=True)  # type: ignore[arg-type]

    assert "VISION–IMU ACQUISITION & PRESENTATION" not in html
    assert '<section class="camera-column">' in html
    assert '<section class="live-column">' in html
    assert html.index('class="camera-column"') < html.index('class="live-column"')
    assert html.index('class="live-column"') < html.index('class="control-panel"')
    assert "grid-template-columns:minmax(0,1.2fr) minmax(0,1fr) minmax(300px,.8fr)" in html
    assert "@media(max-width:1180px)" in html
    assert "@media(max-width:760px)" in html


def test_page_hides_serial_controls_for_bridge_and_renders_recommendations(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    html = create_app(runtime).test_client().get("/").get_data(as_text=True)  # type: ignore[arg-type]

    assert 'id="imuSerialControls"' in html
    assert "imuSerialControls.hidden=bridgeManaged" in html
    assert "else if(!window.imuPortsLoaded) refreshImuPorts()" in html
    assert 'id="imuRecommendationState"' in html
    assert 'id="imuRecommendationValues"' in html
    assert "static_suggestion" in html
    assert "dynamic_suggestion" in html
    assert "recommended_parameters" in html


def test_guided_calibration_is_one_two_stage_session(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    app = create_app(runtime)  # type: ignore[arg-type]
    client = app.test_client()

    html = client.get("/").get_data(as_text=True)
    assert "Start calibration" in html
    assert "Continue to dynamic calibration" in html
    assert "Start 5-second countdown" in html
    assert "Wear both IMUs firmly" in html
    assert 'id="calibrationPath"' in html
    assert "/exports/calibration/" in html

    started = client.post("/api/calibration/start", json={"test_type": "static"})
    assert started.status_code == 200
    assert runtime.calibration_test == "static"

    stopped = client.post("/api/calibration/stop")
    assert stopped.status_code == 200
    assert runtime.calibration_test == ""

    opened = client.post("/api/calibration/open")
    assert opened.status_code == 200
    assert opened.get_json()["path"] == (
        "calibration_suggestions/calibration_run_test"
    )
    assert opened.get_json()["url"].endswith("calibration_run_test/")


def test_export_routes_list_and_serve_owned_files_without_traversal(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    recording = runtime.recording_root / "vision_imu_session_test.zip"
    recording.write_bytes(b"recording")
    run = runtime.calibration_root / "calibration_run_test"
    run.mkdir()
    suggestion = run / "imu_filter_suggestions.yaml"
    suggestion.write_text("status: complete\n", encoding="utf-8")

    client = create_app(runtime).test_client()  # type: ignore[arg-type]

    recording_index = client.get("/exports/recordings/")
    assert recording_index.status_code == 200
    assert "vision_imu_session_test.zip" in recording_index.get_data(as_text=True)
    assert client.get("/exports/recordings/vision_imu_session_test.zip").data == b"recording"

    calibration_index = client.get("/exports/calibration/calibration_run_test/")
    assert calibration_index.status_code == 200
    assert "imu_filter_suggestions.yaml" in calibration_index.get_data(as_text=True)
    suggestion_response = client.get(
        "/exports/calibration/calibration_run_test/imu_filter_suggestions.yaml"
    )
    assert suggestion_response.status_code == 200
    assert b"status: complete" in suggestion_response.data

    assert client.get("/exports/recordings/../secret.txt").status_code == 404
    assert client.get("/exports/calibration/../../secret.txt").status_code == 404


def test_checkerboard_camera_calibration_routes_and_controls(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    client = create_app(runtime).test_client()  # type: ignore[arg-type]
    html = client.get("/").get_data(as_text=True)
    assert "Reference camera calibration" in html
    assert "Start checkerboard calibration" in html
    assert "15 diverse views" in html
    started = client.post("/api/reference/camera-calibration/start")
    assert started.status_code == 200
    assert started.get_json()["state"] == "CAPTURING"
    cancelled = client.post("/api/reference/camera-calibration/cancel")
    assert cancelled.status_code == 200
    assert cancelled.get_json()["state"] == "IDLE"


def test_page_autoscales_dashboard_to_viewport_without_page_scroll(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    html = create_app(runtime).test_client().get("/").get_data(as_text=True)  # type: ignore[arg-type]

    assert 'id="dashboardFit"' in html
    assert "overflow:hidden" in html
    assert "fitDashboardToViewport" in html
    assert "dashboardFit.style.zoom" in html
    assert "scheduleDashboardFit" in html
