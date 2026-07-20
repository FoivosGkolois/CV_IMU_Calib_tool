from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL_URL = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/"
    "tflite/float16/4?lite-format=tflite"
)
MODEL_NAME = "MoveNet SinglePose Thunder TFLite Float16 v4"
DEFAULT_TARGET = Path(__file__).resolve().parent / "models" / "movenet_singlepose_thunder_float16.tflite"
_MIN_MODEL_BYTES = 10_000_000


def download_model(target: Path = DEFAULT_TARGET, *, force: bool = False) -> Path:
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _looks_like_tflite(target) and not force:
        print(f"Pose model already present: {target}")
        return target

    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {MODEL_NAME} to:\n  {target}")
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "Vision-IMU-Calibration/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if not _looks_like_tflite(temporary):
            raise RuntimeError(
                "Downloaded file is not a valid TFLite model. "
                "Check internet access and retry the setup."
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "model_name": MODEL_NAME,
        "source_url": MODEL_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size_bytes": target.stat().st_size,
        "sha256": sha256,
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Model ready ({target.stat().st_size / 1_000_000:.1f} MB, sha256={sha256[:12]}...)")
    return target


def _looks_like_tflite(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < _MIN_MODEL_BYTES:
        return False
    with path.open("rb") as model_file:
        header = model_file.read(8)
    return len(header) == 8 and header[4:8] == b"TFL3"


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Download the pinned {MODEL_NAME} model")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_model(args.target, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
