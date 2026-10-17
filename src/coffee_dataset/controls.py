"""Manual camera controls (white balance, exposure, ...) saved in camera.json.

Auto white balance + a cold white LED ring gives a blue tint. For roast
classification the COLOR of the bean is the signal, so it must be stable and
the same in every photo: lock white balance and exposure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .domain import Camera, Frame


class PropertySetter(Protocol):
    def set(self, prop_id: int, value: float) -> bool: ...


@dataclass(frozen=True)
class CameraControls:
    auto_wb: bool = True
    wb_temperature: int = 4500       # Kelvin; HIGHER = warmer image (less blue)
    auto_exposure: bool = True
    exposure: float = -6.0           # log2 seconds on Windows (-13 dark ... 0 bright)
    brightness: int = 128
    contrast: int = 128
    saturation: int = 128
    sharpness: int = 128
    # Software correction (works even when the camera ignores its own controls)
    gain_b: float = 1.0
    gain_g: float = 1.0
    gain_r: float = 1.0
    gamma: float = 1.0               # >1 darker, <1 brighter
    sharpen: float = 0.0             # software unsharp mask, 0 = off, 0.5-1.0 = clear boost
    autofocus: bool = True
    focus: int = 0                   # manual focus (0-255), only if autofocus is off

    def apply_to(self, cap: PropertySetter) -> None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1 if self.auto_wb else 0)
        if not self.auto_wb:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.wb_temperature)
        # DirectShow: 0.75 = auto exposure, 0.25 = manual
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if self.auto_exposure else 0.25)
        if not self.auto_exposure:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
        cap.set(cv2.CAP_PROP_SATURATION, self.saturation)
        cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)
        if not self.autofocus:
            cap.set(cv2.CAP_PROP_FOCUS, self.focus)

    @property
    def has_software_correction(self) -> bool:
        return self.sharpen > 1e-6 or any(abs(v - 1.0) > 1e-6 for v in (self.gain_b, self.gain_g, self.gain_r, self.gamma))

    def correct(self, frame: Frame) -> Frame:
        """Per-channel gain + gamma, applied to every frame (preview AND saved photos)."""
        if not self.has_software_correction:
            return frame
        x = np.arange(256, dtype=np.float32) / 255.0
        lut = np.empty((256, 1, 3), dtype=np.uint8)
        for channel, gain in enumerate((self.gain_b, self.gain_g, self.gain_r)):
            lut[:, 0, channel] = np.clip(255.0 * np.clip(x * gain, 0.0, 1.0) ** self.gamma, 0, 255).astype(np.uint8)
        out = cv2.LUT(frame, lut)
        if self.sharpen > 1e-6:
            blurred = cv2.GaussianBlur(out, (0, 0), 1.5)
            out = cv2.addWeighted(out, 1.0 + self.sharpen, blurred, -self.sharpen, 0)
        return out

    @staticmethod
    def gains_from_white_patch(patch: Frame) -> tuple[float, float, float]:
        """(blue, green, red) gains that turn the patch into neutral grey."""
        means = patch.reshape(-1, 3).astype(np.float64).mean(axis=0)
        target = means.mean()
        b, g, r = (float(target / max(m, 1.0)) for m in means)
        return b, g, r

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CameraControls | None":
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class ColorCorrectedCamera:
    """Decorator: any Camera, with the software colour correction applied."""

    def __init__(self, inner: Camera, controls: CameraControls) -> None:
        self._inner = inner
        self._controls = controls

    def open(self) -> None:
        self._inner.open()

    def read(self) -> Frame:
        return self._controls.correct(self._inner.read())

    def close(self) -> None:
        self._inner.close()
