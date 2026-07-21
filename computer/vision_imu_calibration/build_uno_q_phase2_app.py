from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


def build_app(project_root: Path, output_dir: Path) -> Path:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    template = (
        project_root
        / "standalone_tools"
        / "vision_imu_calibration"
        / "uno_q_phase2_app_lab_template"
    )
    package_source = project_root / "computer" / "vision_imu_calibration"
    if not template.is_dir() or not package_source.is_dir():
        raise RuntimeError("Run the builder from the CV_IMU_Calib_tool project root")

    ready_dir = output_dir / "uno_q_phase2_app_ready"
    archive_path = output_dir / "uno_q_phase2_app_ready.zip"
    shutil.rmtree(ready_dir, ignore_errors=True)
    archive_path.unlink(missing_ok=True)
    shutil.copytree(template, ready_dir)

    payload_root = ready_dir / "python" / "project"
    computer_root = payload_root / "computer"
    computer_root.mkdir(parents=True, exist_ok=True)
    init_file = project_root / "computer" / "__init__.py"
    if init_file.is_file():
        shutil.copy2(init_file, computer_root / "__init__.py")
    else:
        (computer_root / "__init__.py").write_text("", encoding="utf-8")

    shutil.copytree(
        package_source,
        computer_root / "vision_imu_calibration",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    manifest_lines: list[str] = []
    for path in sorted(p for p in ready_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(ready_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {relative}")
    (ready_dir / "MANIFEST_SHA256.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in ready_dir.rglob("*") if p.is_file()):
            archive.write(path, arcname=f"uno_q_phase2_app_ready/{path.relative_to(ready_dir).as_posix()}")
    return archive_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the UNO Q Phase 2 App Lab payload")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    archive = build_app(args.project_root, args.output_dir)
    print(f"UNO Q Phase 2 App ready: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
