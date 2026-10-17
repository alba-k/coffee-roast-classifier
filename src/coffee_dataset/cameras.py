"""Camera implementations."""

from __future__ import annotations

import sys
import threading
import time

import cv2

from .controls import CameraControls
from .domain import Camera, CameraError, Frame


class OpenCvCamera:
    """Any webcam / UVC camera visible to Windows (laptop cam or the kit cam)."""

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        mirror: bool = False,
        warmup_frames: int = 15,
        fps: float = 15.0,
        controls: CameraControls | None = None,
    ) -> None:
        self._controls = controls
        self._index = index
        self._width = width
        self._height = height
        self._mirror = mirror
        self._warmup = warmup_frames
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap, frame = self._start(compressed=True)
        if frame is None or frame.shape[1] < self._width // 2:
            # The driver ignored our request and gave a tiny image: retry the plain way.
            cap.release()
            cap, frame = self._start(compressed=False)
        self._cap = cap
        if frame is not None:
            height, width = frame.shape[:2]
            print(f"[ .. ] Camera {self._index}: asked {self._width}x{self._height}, got {width}x{height}")
            if width < 640:
                print("[warn] Very low resolution. Close other apps using the camera, or replug it.")

    def _start(self, compressed: bool):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._index, backend)
        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera {self._index}. Run tools/list_cameras.py "
                "to find the right index."
            )
        if compressed:  # compressed video (MJPG) + modest fps: lighter on USB
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._controls is not None:
            self._controls.apply_to(cap)
        frame = None
        for _ in range(self._warmup):  # let auto exposure settle
            ok, got = cap.read()
            if ok and got is not None:
                frame = got
        return cap, frame

    def apply_controls(self, controls: CameraControls) -> None:
        """Change white balance / exposure / ... while the camera is running."""
        if self._cap is not None:
            controls.apply_to(self._cap)

    def open_driver_dialog(self) -> None:
        """Windows/DirectShow: the camera's own settings window (if the driver has one)."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_SETTINGS, 1)

    @property
    def size(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> Frame:
        if self._cap is None:
            raise CameraError("Camera is not open.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Camera returned no image (unplugged?).")
        return cv2.flip(frame, 1) if self._mirror else frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class BackgroundCamera:
    """Reads the camera in a separate thread.

    If the driver is slow or stalls, the window and the keyboard keep working
    (the main loop never waits on the camera). Raises CameraError when no new
    image arrives for `stale_after_s` seconds.
    """

    def __init__(self, inner: Camera, fps: float = 15.0, stale_after_s: float = 5.0) -> None:
        self._inner = inner
        self._period = 1.0 / max(fps, 1.0)
        self._stale = stale_after_s
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._frame: Frame | None = None
        self._error: str | None = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        self._inner.open()
        self._stamp = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="camera-reader", daemon=True)
        self._thread.start()

    def read(self) -> Frame:
        self._new.wait(timeout=0.2)        # never block the window for long
        self._new.clear()
        with self._lock:
            if self._error:
                raise CameraError(self._error)
            frame, stamp = self._frame, self._stamp
        if frame is None or time.monotonic() - stamp > self._stale:
            raise CameraError("The camera stopped sending images (driver busy or unplugged).")
        return frame

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._inner.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                frame = self._inner.read()
            except CameraError as exc:
                with self._lock:
                    self._error = str(exc)
                return
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()
            self._new.set()
            self._stop.wait(max(0.0, self._period - (time.monotonic() - started)))
