"""Entry point: wires the concrete components together."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from coffee_dataset.config import Settings  # noqa: E402
from coffee_dataset.display import OpenCvWindow  # noqa: E402
from coffee_dataset.domain import CameraError  # noqa: E402
from coffee_dataset.factory import build_camera  # noqa: E402
from coffee_dataset.hud import PurpleHud  # noqa: E402
from coffee_dataset.notifiers import BeepNotifier, CompositeNotifier, ConsoleNotifier  # noqa: E402
from coffee_dataset.session import CaptureSession  # noqa: E402
from coffee_dataset.storage import FolderDatasetStore  # noqa: E402
from coffee_dataset.triggers import IntervalTrigger  # noqa: E402


def parse_args() -> Settings:
    d = Settings()
    p = argparse.ArgumentParser(description="Record a coffee roast image dataset.")
    p.add_argument("--camera", type=int, default=d.camera_index, help="camera index (see tools/list_cameras.py)")
    p.add_argument("--interval", type=float, default=d.interval_s, help="seconds between photos")
    p.add_argument("--warn", type=float, default=d.warn_s, help="beep this many seconds before each photo")
    p.add_argument("--target", type=int, default=d.target_per_class, help="photos per class")
    p.add_argument("--output", type=Path, default=d.output_dir, help="dataset folder (photos go to <output>/raw)")
    p.add_argument("--quality", type=int, default=d.jpeg_quality, help="JPEG quality 1-100")
    p.add_argument("--save-width", type=int, default=d.save_width, help="saved photo width in px")
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--height", type=int, default=d.height)
    p.add_argument("--mirror", action="store_true", help="flip the image horizontally")
    p.add_argument("--test", action="store_true", help="practice mode: saves to dataset_test/ with one class 'prueba', 12 photos")
    p.add_argument("--crop-file", type=Path, default=Path("crop.json"), help="area to keep (made by tools/set_crop.py)")
    p.add_argument("--no-crop", action="store_true", help="ignore crop.json")
    p.add_argument("--controls-file", type=Path, default=d.controls_file, help="camera colour settings (tools/tune_camera.py)")
    p.add_argument("--fps", type=float, default=d.fps, help="camera frames per second (lower = lighter)")
    p.add_argument("--label", default=None, help="start with this class selected, e.g. verde")
    p.add_argument("--layout", default=None, help="start with this layout: grupo, disperso or individual")
    p.add_argument("--start", action="store_true", help="start recording right away")
    p.add_argument("--no-beep", action="store_true", help="disable sounds")
    a = p.parse_args()
    if a.test:
        return Settings(
            camera_index=a.camera, width=a.width, height=a.height, mirror=a.mirror,
            interval_s=a.interval, warn_s=a.warn, target_per_class=12,
            jpeg_quality=a.quality, save_width=a.save_width,
            output_dir=Path("dataset_test"), labels=("prueba",), crop_file=a.crop_file,
            use_crop=not a.no_crop, controls_file=a.controls_file, fps=a.fps, beep=not a.no_beep,
        )
    return Settings(
        camera_index=a.camera,
        width=a.width,
        height=a.height,
        mirror=a.mirror,
        interval_s=a.interval,
        warn_s=a.warn,
        target_per_class=a.target,
        jpeg_quality=a.quality,
        save_width=a.save_width,
        output_dir=a.output,
        crop_file=a.crop_file,
        use_crop=not a.no_crop,
        controls_file=a.controls_file,
        fps=a.fps,
        start_label=a.label,
        start_layout=a.layout,
        autostart=a.start,
        beep=not a.no_beep,
    )


def main() -> int:
    s = parse_args()
    if s.start_label is not None and s.start_label not in s.labels:
        print(f"[fail] Unknown class '{s.start_label}'. Choose one of: {', '.join(s.labels)}")
        return 2
    if s.start_layout is not None and s.start_layout not in s.arrangements:
        print(f"[fail] Unknown layout '{s.start_layout}'. Choose one of: {', '.join(s.arrangements)}")
        return 2
    notifiers = [ConsoleNotifier()] + ([BeepNotifier()] if s.beep else [])
    session = CaptureSession(
        settings=s,
        camera=build_camera(s),
        trigger=IntervalTrigger(s.interval_s),
        notifier=CompositeNotifier(notifiers),
        sink=FolderDatasetStore(s.output_dir / "raw", s.labels, s.jpeg_quality, s.save_width),
        display=OpenCvWindow(),
        hud=PurpleHud(),
    )
    try:
        session.run()
    except CameraError as exc:
        print(f"[fail] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
