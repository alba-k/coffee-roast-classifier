"""Abstractions (ports) used by the capture session.

Every concrete piece (camera, trigger, notifier, storage, display) can be
swapped without touching the session logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

Frame = np.ndarray


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class Camera(Protocol):
    def open(self) -> None: ...

    def read(self) -> Frame: ...

    def close(self) -> None: ...


class Trigger(Protocol):
    """Decides WHEN a photo must be taken."""

    def reset(self, now: float) -> None: ...

    def seconds_left(self, now: float) -> float: ...

    def should_fire(self, now: float) -> bool: ...

    def mark_fired(self, now: float) -> None: ...


class Notifier(Protocol):
    """Tells the person what is happening (console, beep, Telegram...)."""

    def info(self, message: str) -> None: ...

    def warn(self, label: str) -> None: ...

    def saved(self, label: str, count: int, target: int) -> None: ...


class DatasetSink(Protocol):
    """Where photos are stored."""

    def save(self, frame: Frame, label: str, tag: str = "") -> Path: ...

    def count(self, label: str) -> int: ...


class Display(Protocol):
    """Shows the live image and returns the pressed key (or -1)."""

    def show(self, frame: Frame) -> int: ...

    def close(self) -> None: ...
