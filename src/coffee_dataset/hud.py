"""Dark / purple / monospace-style heads-up display drawn over the live image.

The overlay is drawn on a COPY: saved photos never contain the HUD.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .domain import Frame

# BGR colours
BG = (24, 14, 14)
ACCENT = (247, 85, 168)      # purple #A855F7
TEXT = (225, 215, 215)
DIM = (150, 135, 135)
OK = (160, 220, 120)
FONT = cv2.FONT_HERSHEY_PLAIN  # closest to a monospace font in OpenCV


@dataclass(frozen=True)
class HudState:
    labels: tuple[str, ...]
    counts: tuple[int, ...]
    current: int
    target: int
    running: bool
    seconds_left: float
    interval_s: float
    arrangement: str = ""
    message: str = ""


class PurpleHud:
    def render(self, frame: Frame, state: HudState) -> Frame:
        """Panel on the LEFT, image on the right (the panel never covers the image)."""
        h = frame.shape[0]
        scale = max(0.8, h / 720)
        line = int(22 * scale)
        panel_w = int(300 * scale)
        panel = np.full((h, panel_w, 3), BG, dtype=np.uint8)
        out = np.hstack([panel, frame])
        cv2.line(out, (panel_w - 2, 0), (panel_w - 2, h), ACCENT, 2)  # inside the panel

        x, y = int(14 * scale), int(30 * scale)
        self._text(out, "COFFEE//DATASET", (x, y), ACCENT, 1.6 * scale)
        y += line + 6
        status = "REC" if state.running else "PAUSED"
        self._text(out, f"[{status}] interval {state.interval_s:g}s", (x, y), TEXT, 1.2 * scale)
        y += line
        self._text(out, f"layout: {state.arrangement}", (x, y), TEXT, 1.2 * scale)
        y += line + 10

        for i, (label, count) in enumerate(zip(state.labels, state.counts), start=1):
            active = (i - 1) == state.current
            done = count >= state.target
            colour = ACCENT if active else (OK if done else DIM)
            marker = ">" if active else " "
            text = f"{marker}{i} {label:<13}{count:>4}/{state.target}"
            self._text(out, text, (x, y), colour, 1.2 * scale)
            y += line

        # countdown bar
        y = h - int(8 * line) - int(70 * scale)
        if state.running:
            frac = 1.0 - min(1.0, state.seconds_left / max(state.interval_s, 1e-6))
            bar_w = panel_w - 2 * x
            cv2.rectangle(out, (x, y), (x + bar_w, y + 10), DIM, 1)
            cv2.rectangle(out, (x, y), (x + int(bar_w * frac), y + 10), ACCENT, -1)
            self._text(out, f"next shot in {state.seconds_left:0.1f}s", (x, y + 30), TEXT, 1.2 * scale)

        keys = (
            "KEYS",
            "1-8    pick class",
            "G      beans together",
            "D      beans spread",
            "I      one single bean",
            "SPACE  start / pause",
            "S      photo now",
            "Q      quit",
        )
        top = h - int(8 * line) - 6
        for j, k in enumerate(keys):
            colour = ACCENT if j == 0 else DIM
            self._text(out, k, (x, top + j * line), colour, 1.1 * scale)

        if state.message:
            self._text(out, state.message, (panel_w + 16, int(30 * scale)), ACCENT, 1.6 * scale)
        return out

    @staticmethod
    def _text(img: np.ndarray, text: str, org: tuple[int, int], colour, scale: float) -> None:
        cv2.putText(img, text, org, FONT, scale, colour, 1, cv2.LINE_AA)
