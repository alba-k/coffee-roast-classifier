"""Automatic first pass of curation: measure every raw frame and reject the clearly bad ones.

Measuring (FrameMetrics) is separate from deciding (Rule objects), so rules can be added,
removed or re-tuned without touching the rest. Raw files are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    coverage: float      # share of the central area that is not white paper (beans, hands, objects...)
    largest_blob: int    # pixels of the biggest bean-sized blob in the central area (0 = none)
    hand_area: float     # share of the whole frame that is a large, smooth, skin-coloured region
    glove_area: float    # share of the whole frame that is a large warm-white region (latex glove)
    sharpness: float     # variance of the Laplacian (low = blurry)
    dark_share: float    # share of non-paper pixels that are very dark (dark beans)
    brightness: float    # mean V of the whole frame


def measure(image: np.ndarray) -> FrameMetrics:
    h, w = image.shape[:2]
    scale = 800 / w if w != 800 else 1.0
    if scale != 1.0:
        image = cv2.resize(image, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    y0, y1, x0, x1 = int(h * 0.12), int(h * 0.88), int(w * 0.15), int(w * 0.85)
    hsv = hsv_full[y0:y1, x0:x1]
    gray = gray_full[y0:y1, x0:x1]
    sat, val = hsv[..., 1].astype(int), hsv[..., 2].astype(int)

    paper = (sat < 55) & (val > 135)
    beans = cv2.morphologyEx((~paper).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    coverage = float(beans.mean())
    count, _, stats, _ = cv2.connectedComponentsWithStats(beans)
    areas = stats[1:, cv2.CC_STAT_AREA]
    sized = areas[(areas > 250) & (areas < 6000)]
    largest = int(sized.max()) if sized.size else 0
    dark = float(((val < 70) & (beans > 0)).sum() / max(1, beans.sum()))

    return FrameMetrics(
        coverage=coverage,
        largest_blob=largest,
        hand_area=_hand_area(image, gray_full),
        glove_area=_glove_area(image),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        dark_share=dark,
        brightness=float(hsv_full[..., 2].mean()),
    )


def _hand_area(image: np.ndarray, gray: np.ndarray) -> float:
    """Skin colour + smooth texture, then an opening wider than any bean: only hands survive."""
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    sat, val = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1], cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    skin = (ycc[..., 1] > 135) & (ycc[..., 1] < 180) & (ycc[..., 2] > 85) & (ycc[..., 2] < 135) & (sat > 25) & (val > 80)
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (7, 7))
    local_std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mean * mean, 0))
    mask = (skin & (local_std < 4)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return float(mask.mean())


def _glove_area(image: np.ndarray) -> float:
    """Latex gloves are bright, slightly warm and a bit saturated; blown-out paper is pure white."""
    b, r = image[..., 0].astype(int), image[..., 2].astype(int)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] > 190) & (hsv[..., 1] >= 8) & (hsv[..., 1] < 45) & (r - b > -3)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return float(mask.mean())


class Rule(Protocol):
    def reject_reason(self, m: FrameMetrics, label: str, layout: str) -> str | None: ...


@dataclass(frozen=True)
class HandRule:
    max_hand: float = 0.025
    max_glove: float = 0.005

    def reject_reason(self, m: FrameMetrics, label: str, layout: str) -> str | None:
        if m.hand_area > self.max_hand:
            return "hand"
        return "glove" if m.glove_area > self.max_glove else None


@dataclass(frozen=True)
class TooFewBeansRule:
    """Group/spread photos need many beans; 'individual' photos need exactly one bean."""
    min_coverage: float = 0.05

    def reject_reason(self, m: FrameMetrics, label: str, layout: str) -> str | None:
        if layout == "individual":
            return None if m.largest_blob > 0 else "no_bean"
        return "few_beans" if m.coverage < self.min_coverage else None


@dataclass(frozen=True)
class BlurRule:
    min_sharpness: float = 25.0

    def reject_reason(self, m: FrameMetrics, label: str, layout: str) -> str | None:
        if layout == "individual":      # an almost empty frame is always "flat", that is not blur
            return None
        return "blurry" if m.sharpness < self.min_sharpness else None


@dataclass(frozen=True)
class ExposureRule:
    min_brightness: float = 90.0
    max_brightness: float = 215.0

    def reject_reason(self, m: FrameMetrics, label: str, layout: str) -> str | None:
        if not self.min_brightness <= m.brightness <= self.max_brightness:
            return "bad_exposure"
        return None


@dataclass(frozen=True)
class MixedRoastRule:
    """Reject frames that contain too many beans darker than the class should have."""
    max_dark_share: dict[str, float]

    def reject_reason(self, m: FrameMetrics, label: str, layout: str) -> str | None:
        limit = self.max_dark_share.get(label)
        return "mixed_roast" if limit is not None and m.dark_share > limit else None


def default_rules() -> tuple[Rule, ...]:
    return (HandRule(), TooFewBeansRule(), BlurRule(), ExposureRule(), MixedRoastRule({"medio": 0.10}))


def judge(m: FrameMetrics, label: str, layout: str, rules: tuple[Rule, ...]) -> str | None:
    """First reason to reject, or None to keep. 'nada_otro' only needs a usable exposure."""
    if label == "nada_otro":
        return ExposureRule(60.0, 250.0).reject_reason(m, label, layout)
    for rule in rules:
        reason = rule.reject_reason(m, label, layout)
        if reason:
            return reason
    return None
