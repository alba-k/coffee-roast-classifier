"""Fix the colour (blue tint) and brightness with live sliders.

W        click-calibrate: drag a box on something that should be WHITE (the box
         floor) and press ENTER. The blue tint is removed automatically.
sliders  fine tune: 'blue' / 'green' / 'red' (100 = no change), 'gamma' (higher = darker),
         'sharpen' (clearer image), 'autofocus' 0 + 'focus' (if the camera allows it).
P        open the camera's own settings window (Windows), if the driver has one.
S/ENTER  save to camera.json (main.py uses it automatically).   Q = quit.

Colour, gamma and sharpen are done in software, so they always work, and they are
applied to every saved photo too. 'autofocus'/'focus' talk to the camera: many cheap
cameras ignore them (then turn the lens ring by hand, or raise the camera).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from coffee_dataset.cameras import OpenCvCamera  # noqa: E402
from coffee_dataset.controls import CameraControls  # noqa: E402
from coffee_dataset.crop import CropRegion  # noqa: E402

WIN = "COFFEE//TUNE"


def noop(_value: int) -> None:
    pass


def read_controls(base: CameraControls) -> CameraControls:
    g = lambda name: cv2.getTrackbarPos(name, WIN)  # noqa: E731
    return dataclasses.replace(
        base,
        gain_b=max(30, g("blue x100")) / 100,
        gain_g=max(30, g("green x100")) / 100,
        gain_r=max(30, g("red x100")) / 100,
        gamma=max(30, g("gamma x100")) / 100,
        sharpen=g("sharpen x100") / 100,
        autofocus=bool(g("autofocus")),
        focus=g("focus"),
    )


def set_software_sliders(c: CameraControls) -> None:
    cv2.setTrackbarPos("blue x100", WIN, round(c.gain_b * 100))
    cv2.setTrackbarPos("green x100", WIN, round(c.gain_g * 100))
    cv2.setTrackbarPos("red x100", WIN, round(c.gain_r * 100))
    cv2.setTrackbarPos("gamma x100", WIN, round(c.gamma * 100))


def calibrate_white(raw_view, current: CameraControls) -> CameraControls | None:
    """Ask for a white area and return controls whose gains make it neutral."""
    print("Drag a box over something WHITE (the box floor), then press ENTER.")
    x, y, w, h = cv2.selectROI("COFFEE//WHITE", raw_view, showCrosshair=False)
    cv2.destroyWindow("COFFEE//WHITE")
    if w == 0 or h == 0:
        return None
    patch = raw_view[y : y + h, x : x + w]
    b, g, r = CameraControls.gains_from_white_patch(patch)
    scale = 1.0 / max(b, g, r)          # never push a channel above 1.0 (no clipping)
    level = float(patch.reshape(-1, 3).mean()) * scale / 255.0     # brightness of the floor after gains
    gamma = min(1.0, max(0.4, math.log(0.8) / math.log(max(level, 0.05))))   # aim for a light-grey floor
    return dataclasses.replace(current, gain_b=round(b * scale, 2), gain_g=round(g * scale, 2),
                               gain_r=round(r * scale, 2), gamma=round(gamma, 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--file", type=Path, default=Path("camera.json"))
    ap.add_argument("--crop-file", type=Path, default=Path("crop.json"))
    a = ap.parse_args()

    start = CameraControls.load(a.file) or CameraControls()
    cam = OpenCvCamera(a.camera, 1280, 720, fps=15.0, controls=start)
    cam.open()
    crop = CropRegion.load(a.crop_file)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 760, 560)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    for name, value, top in (
        ("blue x100", round(start.gain_b * 100), 200),
        ("green x100", round(start.gain_g * 100), 200),
        ("red x100", round(start.gain_r * 100), 200),
        ("gamma x100", round(start.gamma * 100), 250),
        ("sharpen x100", round(start.sharpen * 100), 200),
        ("autofocus", int(start.autofocus), 1),
        ("focus", start.focus, 255),
    ):
        cv2.createTrackbar(name, WIN, max(0, min(top, value)), top, noop)

    last = start
    try:
        while True:
            raw = cam.read()
            current = read_controls(last)
            neutral = dict(gain_b=1.0, gain_g=1.0, gain_r=1.0, gamma=1.0)
            if dataclasses.replace(current, **neutral) != dataclasses.replace(last, **neutral):
                cam.apply_controls(current)       # hardware sliders changed
            last = current
            view = crop.apply(raw) if crop else raw
            shown = current.correct(view).copy()
            cv2.putText(shown, "W white-calibrate  P driver  S save  Q quit", (12, 28),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (247, 85, 168), 1, cv2.LINE_AA)
            cv2.imshow(WIN, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), 13):
                last.save(a.file)
                print(f"Saved {a.file}")
                break
            if key in (ord("q"), 27):
                print("Not saved.")
                break
            if key == ord("p"):
                cam.open_driver_dialog()
            if key == ord("w"):
                fixed = calibrate_white(view.copy(), dataclasses.replace(current, gain_b=1.0, gain_g=1.0, gain_r=1.0, gamma=1.0))
                if fixed is not None:
                    set_software_sliders(fixed)
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
