"""Crop to the useful area (the box floor) so dark edges never reach the dataset.

The region is stored as FRACTIONS of the frame (crop.json), so it keeps working
if the camera resolution changes. It is applied to the camera frames, hence both
the live preview and the saved photos use it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .domain import Camera, Frame


MIN_SIDE = 200  # px; a smaller crop means the region does not fit this camera image


@dataclass(frozen=True)
class CropRegion:
    x: float
    y: float
    w: float
    h: float

    def apply(self, frame: Frame) -> Frame:
        height, width = frame.shape[:2]
        x0, y0 = int(self.x * width), int(self.y * height)
        x1, y1 = int((self.x + self.w) * width), int((self.y + self.h) * height)
        cropped = frame[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
        if cropped.shape[0] < MIN_SIDE or cropped.shape[1] < MIN_SIDE:
            return frame  # never return a tiny image: show the whole frame instead
        return cropped

    @classmethod
    def square_from_pixels(cls, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> "CropRegion":
        """Turn a selected rectangle into the biggest centred SQUARE inside it."""
        side = min(w, h)
        x += (w - side) // 2
        y += (h - side) // 2
        return cls(x / frame_w, y / frame_h, side / frame_w, side / frame_h)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CropRegion | None":
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["x"], data["y"], data["w"], data["h"])


class CroppedCamera:
    """Decorator: any Camera, but every frame is cropped."""

    def __init__(self, inner: Camera, region: CropRegion) -> None:
        self._inner = inner
        self._region = region

    def open(self) -> None:
        self._inner.open()

    def read(self) -> Frame:
        return self._region.apply(self._inner.read())

    def close(self) -> None:
        self._inner.close()
