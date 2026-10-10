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
LIGHTNING_MODEL_URL = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/"
    "tflite/float16/4?lite-format=tflite"
)
LIGHTNING_MODEL_NAME = "MoveNet SinglePose Lightning TFLite Float16 v4"
LIGHTNING_TARGET = Path(__file__).resolve().parent / "models" / "movenet_singlepose_lightning_float16.tflite"
_LIGHTNING_MIN_MODEL_BYTES = 2_000_000
DEFAULT_TARGET = Path(__file__).resolve().parent / "models" / "movenet_singlepose_thunder_float16.tflite"
_MIN_MODEL_BYTES = 10_000_000


def download_model(
    target: Path = DEFAULT_TARGET, *, force: bool = False, variant: str = "thunder"
) -> Path:
    if variant not in ("thunder", "lightning"):
        raise ValueError("variant must be thunder or lightning")
    url = LIGHTNING_MODEL_URL if variant == "lightning" else MODEL_URL
    name = LIGHTNING_MODEL_NAME if variant == "lightning" else MODEL_NAME
    min_bytes = _LIGHTNING_MIN_MODEL_BYTES if variant == "lightning" else _MIN_MODEL_BYTES
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _looks_like_tflite(target, min_bytes=min_bytes) and not force:
        print(f"Pose model already present: {target}")
        return target

    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {name} to:\n  {target}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Vision-IMU-Calibration/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if not _looks_like_tflite(temporary, min_bytes=min_bytes):
            raise RuntimeError(
                "Downloaded file is not a valid TFLite model. "
                "Check internet access and retry the setup."
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "model_name": name,
        "source_url": url,
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


def _looks_like_tflite(path: Path, *, min_bytes: int = _MIN_MODEL_BYTES) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as model_file:
        header = model_file.read(8)
    return len(header) == 8 and header[4:8] == b"TFL3"


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Download the pinned {MODEL_NAME} model")
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--variant", choices=("thunder", "lightning"), default="thunder")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = args.target or (LIGHTNING_TARGET if args.variant == "lightning" else DEFAULT_TARGET)
    download_model(target, force=args.force, variant=args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
