"""Print which camera indexes work, so you can choose the kit camera."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from coffee_dataset.cameras import OpenCvCamera  # noqa: E402
from coffee_dataset.domain import CameraError  # noqa: E402


def main() -> None:
    found = False
    for index in range(6):
        cam = OpenCvCamera(index, 1280, 720, warmup_frames=3)
        try:
            cam.open()
            frame = cam.read()
            h, w = frame.shape[:2]
            print(f"[ ok ] index {index}: {w}x{h}")
            cv2.imwrite(f"camera_{index}.jpg", frame)
            found = True
        except CameraError:
            print(f"[ -- ] index {index}: not available")
        finally:
            cam.close()
    if found:
        print("Open camera_<n>.jpg to see which one is the box camera.")


if __name__ == "__main__":
    main()
