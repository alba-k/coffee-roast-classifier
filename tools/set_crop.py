"""Pick the area to keep (the box floor). Drag a rectangle with the mouse.

ENTER or SPACE = save, C = cancel. The saved area is turned into a square and
stored in crop.json; main.py uses it automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from coffee_dataset.cameras import OpenCvCamera  # noqa: E402
from coffee_dataset.crop import CropRegion  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--file", type=Path, default=Path("crop.json"))
    ap.add_argument("--mirror", action="store_true")
    a = ap.parse_args()

    cam = OpenCvCamera(a.camera, 1280, 720, a.mirror)
    cam.open()
    try:
        frame = cam.read()
    finally:
        cam.close()

    h, w = frame.shape[:2]
    print("Drag a rectangle over the box floor, then press ENTER (C = cancel).")
    x, y, rw, rh = cv2.selectROI("COFFEE//CROP", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if rw == 0 or rh == 0:
        print("Nothing selected. crop.json not changed.")
        return 1
    region = CropRegion.square_from_pixels(x, y, rw, rh, w, h)
    region.save(a.file)
    cropped = region.apply(frame)
    cv2.imwrite("crop_preview.jpg", cropped)
    print(f"Saved {a.file}  ->  square {cropped.shape[1]}x{cropped.shape[0]} px. Preview: crop_preview.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
