"""Storage: one folder per class + manifest.csv (proof of manual recording)."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2

from .domain import Frame

MANIFEST_HEADER = ("file", "label", "arrangement", "timestamp", "width", "height", "kb")


class FolderDatasetStore:
    """<root>/<label>/<label>_<arrangement>_YYYYmmdd_HHMMSS_mmm.jpg

    <root> is the RAW stage (dataset/raw). Photos are downscaled to
    max_width so they stay light for Teachable Machine.
    """

    def __init__(
        self,
        root: Path,
        labels: tuple[str, ...],
        jpeg_quality: int = 85,
        max_width: int = 640,
    ) -> None:
        self._root = Path(root)
        self._quality = jpeg_quality
        self._max_width = max_width
        self._manifest = self._root / "manifest.csv"
        for label in labels:
            (self._root / label).mkdir(parents=True, exist_ok=True)
        if not self._manifest.exists():
            with self._manifest.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(MANIFEST_HEADER)

    def save(self, frame: Frame, label: str, tag: str = "") -> Path:
        now = datetime.now()
        middle = f"_{tag}" if tag else ""
        name = f"{label}{middle}_{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}.jpg"
        frame = self._shrink(frame)
        path = self._root / label / name
        n = 1
        while path.exists():  # never overwrite a photo taken in the same millisecond
            name = f"{path.stem.rsplit('_dup', 1)[0]}_dup{n}.jpg"
            path = self._root / label / name
            n += 1
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            raise OSError("Could not encode frame as JPEG.")
        path.write_bytes(buffer.tobytes())  # works with any Windows path
        height, width = frame.shape[:2]
        with self._manifest.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                (
                    f"{label}/{name}", label, tag, now.isoformat(timespec="milliseconds"),
                    width, height, round(len(buffer) / 1024, 1),
                )
            )
        return path

    def _shrink(self, frame: Frame) -> Frame:
        height, width = frame.shape[:2]
        if width <= self._max_width:
            return frame
        new_h = round(height * self._max_width / width)
        return cv2.resize(frame, (self._max_width, new_h), interpolation=cv2.INTER_AREA)

    def count(self, label: str) -> int:
        folder = self._root / label
        return sum(1 for _ in folder.glob("*.jpg")) if folder.exists() else 0
