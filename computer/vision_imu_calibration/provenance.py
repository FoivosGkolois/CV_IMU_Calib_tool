from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .final_protocol import FINAL_PROTOCOL_VERSION


def load_runtime_provenance() -> dict[str, Any]:
    """Load immutable build metadata when running from an App Lab package.

    PC/source-tree runs remain supported and are reported explicitly as such.
    """
    candidates = [
        Path.cwd() / "BUILD_INFO.json",
        Path(__file__).resolve().parents[2] / "BUILD_INFO.json",
    ]
    build_info: dict[str, Any] = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict):
            build_info = raw
            break
    return {
        "protocol_version": FINAL_PROTOCOL_VERSION,
        "build_info": build_info,
        "build_info_available": bool(build_info),
    }
