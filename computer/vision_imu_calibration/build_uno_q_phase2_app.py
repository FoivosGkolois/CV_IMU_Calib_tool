from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
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

    build_info = _build_info(project_root, package_source)
    (payload_root / "BUILD_INFO.json").write_text(
        json.dumps(build_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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



def _git_value(project_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return ""
    return result.stdout.strip()



def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return ""
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

def _build_info(project_root: Path, package_source: Path) -> dict[str, object]:
    filter_path = package_source / "imu_filter.py"
    protocol_path = package_source / "final_protocol.py"
    sketch_path = project_root / "standalone_tools" / "vision_imu_calibration" / "uno_q_phase2_app_lab_template" / "sketch" / "sketch.ino"
    commit = _git_value(project_root, "rev-parse", "HEAD")
    branch = _git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git_value(project_root, "status", "--porcelain"))
    return {
        "schema_version": 1,
        "built_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "filter_implementation_file": "computer/vision_imu_calibration/imu_filter.py",
        "filter_implementation_sha256": (
            hashlib.sha256(filter_path.read_bytes()).hexdigest() if filter_path.is_file() else ""
        ),
        "protocol_file_sha256": (
            hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.is_file() else ""
        ),
        "calibration_source_tree_sha256": _tree_digest(package_source),
        "mcu_sketch_sha256": (
            hashlib.sha256(sketch_path.read_bytes()).hexdigest() if sketch_path.is_file() else ""
        ),
    }

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
