"""Dark / purple heads-up display for the live classifier (drawn on a copy of the frame)."""

from __future__ import annotations

import cv2
import numpy as np

from .domain import Image, Prediction

BG = (24, 14, 14)
ACCENT = (247, 85, 168)   # purple #A855F7 (BGR)
TEXT = (225, 215, 215)
DIM = (150, 135, 135)
FONT = cv2.FONT_HERSHEY_PLAIN

# Order of the bars in the panel (the model's own output order is not changed).
DISPLAY_ORDER = ("Verde", "Claro", "Medio", "MedioOscuro", "Oscuro", "Quemado", "Fondo")


def display_rows(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Scores sorted by DISPLAY_ORDER; classes not in the list go last, in their original order."""
    rank = {name: i for i, name in enumerate(DISPLAY_ORDER)}
    return sorted(scores.items(), key=lambda kv: rank.get(kv[0], len(rank)))


class PurpleOverlay:
    def render(self, frame: Image, prediction: Prediction, shown_label: str, area: tuple[int, int, int, int]) -> Image:
        """Panel on the left with one bar per class; a purple square marks what the model sees."""
        h = frame.shape[0]
        scale = max(0.8, h / 720)
        panel_w = int(320 * scale)
        panel = np.full((h, panel_w, 3), BG, dtype=np.uint8)
        view = frame.copy()
        x0, y0, x1, y1 = area
        cv2.rectangle(view, (x0, y0), (x1, y1), ACCENT, 2)
        out = np.hstack([panel, view])
        cv2.line(out, (panel_w - 2, 0), (panel_w - 2, h), ACCENT, 2)

        x, y = int(14 * scale), int(30 * scale)
        cv2.putText(out, "COFFEE//CLASSIFIER", (x, y), FONT, 1.5 * scale, ACCENT, 2, cv2.LINE_AA)
        y += int(46 * scale)
        cv2.putText(out, shown_label.upper(), (x, y), FONT, 2.4 * scale, TEXT, 2, cv2.LINE_AA)
        y += int(36 * scale)
        bar_w = panel_w - 2 * x
        for name, score in display_rows(prediction.scores):
            best = name == prediction.label
            colour = ACCENT if best else DIM
            cv2.putText(out, f"{name:<12}{score:>4.0%}", (x, y), FONT, 1.2 * scale, colour, 1, cv2.LINE_AA)
            y += int(8 * scale)
            cv2.rectangle(out, (x, y), (x + bar_w, y + int(8 * scale)), (50, 40, 40), -1)
            cv2.rectangle(out, (x, y), (x + int(bar_w * score), y + int(8 * scale)), colour, -1)
            y += int(26 * scale)
        cv2.putText(out, "[S] save   [Q] quit", (x, h - int(16 * scale)), FONT, 1.1 * scale, DIM, 1, cv2.LINE_AA)
        return out
