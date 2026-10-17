"""Central settings. Change defaults here or with command line flags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Class names are DATA (they become the folder names Teachable Machine will use).
DEFAULT_LABELS: tuple[str, ...] = (
    "verde",
    "claro",
    "medio",
    "medio_oscuro",
    "oscuro",
    "quemado",
    "defectuoso",
    "nada_otro",
)


# Plain-language help shown in the console (what each class means).
CLASS_HELP: dict[str, str] = {
    "verde": "green, raw bean (not roasted)",
    "claro": "light brown (roasted a little)",
    "medio": "medium brown",
    "medio_oscuro": "dark brown, almost no shine",
    "oscuro": "very dark brown, oily shine",
    "quemado": "black / charred (too much roast)",
    "defectuoso": "broken, moldy or uneven beans",
    "nada_otro": "empty floor, or any other object",
}
ARRANGEMENT_HELP: dict[str, str] = {
    "grupo": "beans TOGETHER in a pile",
    "disperso": "beans SPREAD, not touching",
    "individual": "ONE single bean",
}

# How the beans are laid out in the shot (part of the file name and manifest).
ARRANGEMENTS: tuple[str, ...] = ("grupo", "disperso", "individual")
ARRANGEMENT_KEYS: str = "gdi"   # key for each arrangement, same order


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = False
    interval_s: float = 3.0      # seconds between photos
    warn_s: float = 1.0          # warning beep this many seconds before the shot
    target_per_class: int = 100   # 100 shots x 3 s = 5 min per class
    jpeg_quality: int = 92
    save_width: int = 800         # saved photos are never wider than this (light files)
    output_dir: Path = Path("dataset")
    labels: tuple[str, ...] = DEFAULT_LABELS
    arrangements: tuple[str, ...] = ARRANGEMENTS
    crop_file: Path = Path("crop.json")
    use_crop: bool = True
    controls_file: Path = Path("camera.json")   # made by tools/tune_camera.py
    fps: float = 15.0            # frames read per second (lower = lighter for the laptop)
    start_label: str | None = None       # preselected class (--label)
    start_layout: str | None = None      # preselected layout (--layout)
    autostart: bool = False              # begin recording right away (--start)
    beep: bool = True
