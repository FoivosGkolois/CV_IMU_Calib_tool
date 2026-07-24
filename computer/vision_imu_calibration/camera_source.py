from __future__ import annotations

import platform
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

CameraSelector = int | str


class OpenCvCameraSource:
    """OpenCV camera/video adapter with explicit-index and auto-detect modes."""

    def __init__(
        self,
        *,
        camera_index: CameraSelector | None = "auto",
        video_path: str | Path | None = None,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        mirror: bool = True,
        backend: str = "auto",
        cv2_module: Any | None = None,
        platform_name: str | None = None,
    ) -> None:
        if (camera_index is None) == (video_path is None):
            raise ValueError("Provide exactly one of camera_index or video_path")
        self.camera_index = self._normalize_camera_selector(camera_index)
        self.video_path = Path(video_path).expanduser().resolve() if video_path is not None else None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.mirror = bool(mirror)
        self.backend = self._normalize_backend(backend)
        self._cv2 = cv2_module
        self._platform_name = platform_name or platform.system()
        self._capture: Any | None = None
        self._capture_lock = threading.RLock()
        self._prefetched_frame: Any | None = None
        self._active_backend_name: str | None = None
        self._active_camera_index: int | None = None
        self._active_pixel_format: str | None = None

    @property
    def description(self) -> str:
        if self.video_path is not None:
            return str(self.video_path)
        selected = self._active_camera_index
        selector = selected if selected is not None else self.camera_index
        details: list[str] = []
        if self._active_backend_name:
            details.append(self._active_backend_name)
        if self._active_pixel_format:
            details.append(self._active_pixel_format)
        suffix = f" [{', '.join(details)}]" if details else ""
        return f"camera:{selector}{suffix}"

    @property
    def active_camera_index(self) -> int | None:
        return self._active_camera_index

    def open(self) -> None:
        with self._capture_lock:
            self._open_unlocked()

    def read(self) -> Any | None:
        with self._capture_lock:
            if self._capture is None:
                raise RuntimeError("Camera source is not open")

            frame, self._prefetched_frame = self._prefetched_frame, None
            if frame is None:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    if self.video_path is not None:
                        self._capture.set(self._load_cv2().CAP_PROP_POS_FRAMES, 0)
                        ok, frame = self._capture.read()
                    if not ok or frame is None:
                        return None

            if self.mirror:
                frame = self._load_cv2().flip(frame, 1)
            return frame

    def close(self) -> None:
        with self._capture_lock:
            self._close_unlocked()

    def reconfigure(self, *, width: int, height: int, fps: float | None = None) -> None:
        """Reopen a live camera with a new profile and roll back on failure."""
        if self.video_path is not None:
            raise RuntimeError("Camera profile changes are unavailable for video-file replay")
        new_width = int(width)
        new_height = int(height)
        new_fps = self.fps if fps is None else float(fps)
        if new_width <= 0 or new_height <= 0 or new_fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive")

        with self._capture_lock:
            previous = (self.width, self.height, self.fps)
            self.width, self.height, self.fps = new_width, new_height, new_fps
            try:
                self._open_unlocked()
            except Exception:
                self.width, self.height, self.fps = previous
                try:
                    self._open_unlocked()
                except Exception:
                    pass
                raise

    def profile(self) -> dict[str, float | int]:
        with self._capture_lock:
            return {"width": self.width, "height": self.height, "fps": self.fps}

    def _open_unlocked(self) -> None:
        cv2 = self._load_cv2()
        self._close_unlocked()

        if self.video_path is not None:
            if not self.video_path.is_file():
                raise RuntimeError(f"Video file does not exist: {self.video_path}")
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open {self.description}")
            self._capture = capture
            self._active_backend_name = "FILE"
            return

        failures: list[str] = []
        camera_indices = self._camera_index_candidates()
        for camera_index in camera_indices:
            for backend_name, backend_id in self._camera_backend_candidates(cv2):
                for pixel_format in self._pixel_format_candidates():
                    capture = self._create_camera_capture(cv2, camera_index, backend_id)
                    attempt_name = self._attempt_name(camera_index, backend_name, pixel_format)
                    if not capture.isOpened():
                        capture.release()
                        failures.append(f"{attempt_name}: open failed")
                        continue

                    self._configure_camera(cv2, capture, pixel_format=pixel_format)
                    frame = self._read_initial_frame(capture)
                    if frame is None:
                        capture.release()
                        failures.append(f"{attempt_name}: opened but returned no frame")
                        continue

                    self._capture = capture
                    self._prefetched_frame = frame
                    self._active_backend_name = backend_name
                    self._active_camera_index = camera_index
                    self._active_pixel_format = pixel_format
                    return

        attempted = "; ".join(failures) if failures else "no camera devices were discovered"
        selector = self.camera_index
        raise RuntimeError(
            f"Could not open camera:{selector}. Tried {attempted}. "
            f"{self._camera_troubleshooting_hint()}"
        )

    def _close_unlocked(self) -> None:
        capture, self._capture = self._capture, None
        self._prefetched_frame = None
        self._active_backend_name = None
        self._active_camera_index = None
        self._active_pixel_format = None
        if capture is not None:
            capture.release()

    def _camera_index_candidates(self) -> list[int]:
        if isinstance(self.camera_index, int):
            return [self.camera_index]

        discovered = self._discover_camera_indices()
        if discovered:
            return discovered

        # Fallback for platforms without /dev/video* enumeration.
        return list(range(5))

    def _discover_camera_indices(self) -> list[int]:
        if self._platform_name != "Linux":
            return []

        indices: set[int] = set()
        for path in Path("/dev").glob("video*"):
            match = re.fullmatch(r"video(\d+)", path.name)
            if match:
                indices.add(int(match.group(1)))

        def priority(index: int) -> tuple[int, int]:
            label = self._linux_video_label(index).lower()
            preferred = any(token in label for token in ("uvc", "webcam", "camera"))
            internal_codec = any(token in label for token in ("venus", "codec", "encoder", "decoder"))
            if preferred and not internal_codec:
                group = 0
            elif internal_codec:
                group = 2
            else:
                group = 1
            return group, index

        return sorted(indices, key=priority)

    @staticmethod
    def _linux_video_label(index: int) -> str:
        name_file = Path(f"/sys/class/video4linux/video{index}/name")
        try:
            return name_file.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _camera_backend_candidates(self, cv2: Any) -> list[tuple[str, int | None]]:
        available: dict[str, int | None] = {"DEFAULT": None}
        for name, attribute in (
            ("DSHOW", "CAP_DSHOW"),
            ("MSMF", "CAP_MSMF"),
            ("V4L2", "CAP_V4L2"),
        ):
            backend_id = getattr(cv2, attribute, None)
            if backend_id is not None:
                available[name] = int(backend_id)

        if self.backend != "AUTO":
            if self.backend not in available:
                raise RuntimeError(
                    f"Requested camera backend {self.backend} is unavailable in this OpenCV build"
                )
            return [(self.backend, available[self.backend])]

        if self._platform_name == "Windows":
            order = ("DSHOW", "MSMF", "DEFAULT")
        elif self._platform_name == "Linux":
            order = ("V4L2", "DEFAULT")
        else:
            order = ("DEFAULT",)
        return [(name, available[name]) for name in order if name in available]

    def _pixel_format_candidates(self) -> list[str | None]:
        if self._platform_name == "Linux":
            # Most UVC webcams support MJPG, but retain the driver default as fallback.
            return ["MJPG", None]
        return [None]

    @staticmethod
    def _normalize_camera_selector(camera_index: CameraSelector | None) -> CameraSelector | None:
        if camera_index is None:
            return None
        if isinstance(camera_index, bool):
            raise ValueError("camera_index must be a non-negative integer or 'auto'")
        if isinstance(camera_index, int):
            if camera_index < 0:
                raise ValueError("camera_index must be a non-negative integer or 'auto'")
            return camera_index
        normalized = str(camera_index).strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            parsed = int(normalized)
        except ValueError as exc:
            raise ValueError("camera_index must be a non-negative integer or 'auto'") from exc
        if parsed < 0:
            raise ValueError("camera_index must be a non-negative integer or 'auto'")
        return parsed

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = str(backend).strip().upper()
        aliases = {
            "AUTO": "AUTO",
            "DSHOW": "DSHOW",
            "DIRECTSHOW": "DSHOW",
            "MSMF": "MSMF",
            "MEDIAFOUNDATION": "MSMF",
            "V4L2": "V4L2",
            "DEFAULT": "DEFAULT",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            allowed = "auto, dshow, msmf, v4l2, default"
            raise ValueError(f"Unsupported camera backend {backend!r}; choose {allowed}") from exc

    @staticmethod
    def _create_camera_capture(cv2: Any, camera_index: int, backend_id: int | None) -> Any:
        if backend_id is None:
            return cv2.VideoCapture(camera_index)
        return cv2.VideoCapture(camera_index, backend_id)

    def _configure_camera(
        self,
        cv2: Any,
        capture: Any,
        *,
        pixel_format: str | None,
    ) -> None:
        if (
            pixel_format
            and hasattr(cv2, "CAP_PROP_FOURCC")
            and callable(getattr(cv2, "VideoWriter_fourcc", None))
        ):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    @staticmethod
    def _attempt_name(
        camera_index: int,
        backend_name: str,
        pixel_format: str | None,
    ) -> str:
        format_suffix = f"/{pixel_format}" if pixel_format else ""
        return f"camera:{camera_index}/{backend_name}{format_suffix}"

    def _camera_troubleshooting_hint(self) -> str:
        if self._platform_name == "Windows":
            return (
                "Close other camera applications, verify Windows camera privacy permissions, "
                "or pass an explicit index such as --camera 1."
            )
        if self._platform_name == "Linux":
            return (
                "Verify that a UVC capture node is present under /dev/video*, "
                "or pass an explicit index such as --camera 1."
            )
        return "Close other camera applications or pass an explicit camera index."

    @staticmethod
    def _read_initial_frame(capture: Any, attempts: int = 6, delay_s: float = 0.03) -> Any | None:
        for attempt in range(attempts):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            if attempt + 1 < attempts:
                time.sleep(delay_s)
        return None

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCV is not installed in the active Python environment"
                ) from exc
            self._cv2 = cv2
        return self._cv2
