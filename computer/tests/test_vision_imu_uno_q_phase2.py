from __future__ import annotations

from pathlib import Path
import json
import zipfile

from computer.vision_imu_calibration.__main__ import build_parser
from computer.vision_imu_calibration.build_uno_q_phase2_app import build_app


def test_cli_accepts_uno_q_bridge_transport_without_serial_port() -> None:
    args = build_parser().parse_args(["--imu-transport", "uno-q-bridge"])
    assert args.imu_transport == "uno-q-bridge"
    assert args.imu_port is None
    assert args.camera == "auto"


def test_phase2_template_uses_headless_opencv_contrib_for_app_lab_container() -> None:
    project_root = Path(__file__).resolve().parents[2]
    requirements_path = (
        project_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
        / "python"
        / "requirements.txt"
    )
    requirements = {
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "opencv-contrib-python-headless>=4.10,<5" in requirements
    assert "opencv-python-headless>=4.10,<5" not in requirements
    assert "ai-edge-litert==2.1.6" not in requirements


def test_phase2_builder_creates_self_contained_app_payload(tmp_path: Path) -> None:
    project = tmp_path / "project"
    template = (
        project
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
    )
    (template / "python").mkdir(parents=True)
    (template / "sketch").mkdir(parents=True)
    (template / "app.yaml").write_text("name: test\n", encoding="utf-8")
    (template / "python" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (template / "sketch" / "sketch.ino").write_text("void setup(){}\n", encoding="utf-8")

    package = project / "computer" / "vision_imu_calibration"
    package.mkdir(parents=True)
    (project / "computer" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "module.pyc").write_bytes(b"ignored")

    archive = build_app(project, tmp_path / "out")

    assert archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "uno_q_phase2_app_ready/app.yaml" in names
    assert (
        "uno_q_phase2_app_ready/python/project/computer/vision_imu_calibration/module.py"
        in names
    )
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    with zipfile.ZipFile(archive) as bundle:
        build_info = json.loads(
            bundle.read("uno_q_phase2_app_ready/python/project/BUILD_INFO.json").decode("utf-8")
        )
    assert build_info["calibration_source_tree_sha256"]
    assert build_info["mcu_sketch_sha256"]


def test_cli_accepts_explicit_camera_index_and_v4l2_backend() -> None:
    args = build_parser().parse_args(["--camera", "3", "--camera-backend", "v4l2"])
    assert args.camera == 3
    assert args.camera_backend == "v4l2"


def test_phase2_launcher_defaults_to_camera_auto_detection() -> None:
    project_root = Path(__file__).resolve().parents[2]
    launcher_path = (
        project_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
        / "python"
        / "main.py"
    )
    launcher = launcher_path.read_text(encoding="utf-8")

    assert 'os.environ.get("VISION_IMU_CAMERA_INDEX", "auto")' in launcher
    assert 'os.environ.get("VISION_IMU_CAMERA_BACKEND", "auto")' in launcher
    assert 'os.environ.get("VISION_IMU_CAMERA_WIDTH", "640")' in launcher
    assert 'os.environ.get("VISION_IMU_CAMERA_HEIGHT", "480")' in launcher
    assert 'os.environ.get("VISION_IMU_CAMERA_FPS", "30")' in launcher


def test_cli_uses_project_owned_non_configurable_output_roots() -> None:
    args = build_parser().parse_args([])
    assert not hasattr(args, "recording_root")
    assert not hasattr(args, "calibration_root")


def test_phase2_launcher_does_not_override_project_output_roots() -> None:
    project_root = Path(__file__).resolve().parents[2]
    launcher_path = (
        project_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
        / "python"
        / "main.py"
    )
    launcher = launcher_path.read_text(encoding="utf-8")

    assert "--recording-root" not in launcher
    assert "--calibration-root" not in launcher
    assert "VISION_IMU_DATA_ROOT" not in launcher


def test_phase2_launcher_reports_copy_paste_ready_url_and_startup_failure() -> None:
    project_root = Path(__file__).resolve().parents[2]
    template_root = (
        project_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
    )
    launcher = (template_root / "python" / "main.py").read_text(encoding="utf-8")
    readme = (template_root / "README.md").read_text(encoding="utf-8")
    descriptor = (template_root / "app.yaml").read_text(encoding="utf-8")

    assert "WEB GUI READY" in launcher
    assert "COPY/PASTE THE LINK THAT MATCHES THE UNO Q IP" in launcher
    assert "WEB GUI DID NOT START" in launcher
    assert "WEB GUI STARTUP TIMEOUT" in launcher
    assert "socket.create_connection" in launcher
    assert '"--host",' in launcher and '"0.0.0.0",' in launcher
    assert "ports:" in descriptor and "8765" in descriptor
    assert "WEB GUI READY" in readme
    assert "http://<UNO-Q-IP>:8765" in readme
    assert "172.x.x.x" in readme

def test_phase2_sketch_matches_project_router_baud() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sketch_path = (
        project_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
        / "sketch"
        / "sketch.ino"
    )
    sketch = sketch_path.read_text(encoding="utf-8")

    assert "constexpr unsigned long BRIDGE_BAUD = 921600UL;" in sketch
    assert "Bridge.begin(BRIDGE_BAUD)" in sketch
    assert "Bridge.begin();" not in sketch



def test_phase2_launcher_uses_reserved_persistent_data_directory_for_reference_setup() -> None:
    project_root = Path(__file__).resolve().parents[2]
    launcher_path = (
        project_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
        / "python"
        / "main.py"
    )
    launcher = launcher_path.read_text(encoding="utf-8")

    assert '"--reference-setup-root",' in launcher
    assert 'APP_PYTHON_ROOT.parent / "data" / "reference_setup"' in launcher
