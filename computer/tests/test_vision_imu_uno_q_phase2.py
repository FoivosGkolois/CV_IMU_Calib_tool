from __future__ import annotations

from pathlib import Path
import zipfile

from computer.vision_imu_calibration.__main__ import build_parser
from computer.vision_imu_calibration.build_uno_q_phase2_app import build_app


def test_cli_accepts_uno_q_bridge_transport_without_serial_port() -> None:
    args = build_parser().parse_args(["--imu-transport", "uno-q-bridge"])
    assert args.imu_transport == "uno-q-bridge"
    assert args.imu_port is None


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
