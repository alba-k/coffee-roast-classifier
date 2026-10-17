"""Crop every curated photo to the well-lit centre of the box (removes the dark vignette).

Photos whose beans were all outside the crop are dropped, so no empty photo ends up
labelled as a roast level.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .autocurate import TooFewBeansRule, measure
from .crop import CropRegion

PAPER_GREY = 200  # fills the part outside the crop when checking "are there still beans inside?"


def centred_square(frame_w: int, frame_h: int, side_fraction: float = 1.0) -> CropRegion:
    """Biggest centred square (as a fraction of the frame height), e.g. 450x450 of a 800x450 frame."""
    side = int(min(frame_w, frame_h) * side_fraction)
    return CropRegion.square_from_pixels((frame_w - side) // 2, (frame_h - side) // 2, side, side, frame_w, frame_h)


def beans_still_visible(image: np.ndarray, region: CropRegion, layout: str) -> bool:
    """Run the 'enough beans' rule on the frame with everything outside the crop painted as paper."""
    h, w = image.shape[:2]
    x0, y0 = int(region.x * w), int(region.y * h)
    x1, y1 = int((region.x + region.w) * w), int((region.y + region.h) * h)
    masked = np.full_like(image, PAPER_GREY)
    masked[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    metrics = measure(masked)
    return TooFewBeansRule().reject_reason(metrics, "", layout) is None


@dataclass
class CropReport:
    written: dict[str, int]
    dropped: list[tuple[str, str]]


def layout_of(label: str, name: str) -> str:
    rest = name[len(label) + 1:] if name.startswith(label + "_") else name
    return rest.split("_")[0]


def crop_curated(curated: Path, out: Path, labels: tuple[str, ...], region: CropRegion | None = None,
                 quality: int = 92) -> CropReport:
    written: dict[str, int] = {}
    dropped: list[tuple[str, str]] = []
    for label in labels:
        src = Path(curated) / label
        written[label] = 0
        if not src.is_dir():
            continue
        dest = Path(out) / "curated" / label
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(src.glob("*.jpg")):
            if (dest / path.name).exists():          # done in an earlier (interrupted) run
                written[label] += 1
                continue
            image = cv2.imread(str(path))
            if image is None:
                dropped.append((f"{label}/{path.name}", "unreadable"))
                continue
            h, w = image.shape[:2]
            area = region or centred_square(w, h)
            if label != "nada_otro" and not beans_still_visible(image, area, layout_of(label, path.name)):
                dropped.append((f"{label}/{path.name}", "no_beans_in_crop"))
                continue
            cropped = area.apply(image)
            cv2.imwrite(str(dest / path.name), cropped, [cv2.IMWRITE_JPEG_QUALITY, quality])
            written[label] += 1
    return CropReport(written, dropped)


def write_dropped(report: CropReport, path: Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("file", "reason"))
        writer.writerows(report.dropped)
