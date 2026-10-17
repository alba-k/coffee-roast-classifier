"""Builds the camera exactly as the capture tool does (so live use sees the same image)."""

from __future__ import annotations

from .cameras import BackgroundCamera, OpenCvCamera
from .config import Settings
from .controls import CameraControls, ColorCorrectedCamera
from .crop import CroppedCamera, CropRegion


def build_camera(s: Settings):
    controls = CameraControls.load(s.controls_file)
    if controls is not None:
        print(f"[ .. ] Camera colour settings from {s.controls_file}")
    camera = BackgroundCamera(
        OpenCvCamera(s.camera_index, s.width, s.height, s.mirror, fps=s.fps, controls=controls), fps=s.fps
    )
    if controls is not None and controls.has_software_correction:
        print("[ .. ] Software colour correction ON (from camera.json)")
        camera = ColorCorrectedCamera(camera, controls)
    region = CropRegion.load(s.crop_file) if s.use_crop else None
    if region is None:
        print("[ .. ] No crop.json: using the full image. Run tools/set_crop.py to crop.")
        return camera
    print(f"[ .. ] Cropping with {s.crop_file}")
    return CroppedCamera(camera, region)
