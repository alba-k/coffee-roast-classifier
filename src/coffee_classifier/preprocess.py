"""Turn a camera frame / photo into what the model was trained with.

Training photos are the centred square of an 800 px wide frame (450x450), and
Teachable Machine itself feeds 224x224 images scaled to [-1, 1].
"""

from __future__ import annotations

import cv2
import numpy as np

from .domain import Image

INPUT_SIZE = 224
SAVE_WIDTH = 800   # photos were saved at most 800 px wide (Settings.save_width)


def match_training_frame(frame: Image, save_width: int = SAVE_WIDTH) -> Image:
    """Live frames only: same size the dataset photos had before they were cropped."""
    h, w = frame.shape[:2]
    if w <= save_width:
        return frame
    return cv2.resize(frame, (save_width, round(h * save_width / w)), interpolation=cv2.INTER_AREA)


def centre_square(image: Image) -> Image:
    h, w = image.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    return image[y : y + side, x : x + side]


def to_model_input(image: Image) -> np.ndarray:
    """BGR image -> float32 array (1, 224, 224, 3), RGB, values in [-1, 1]."""
    rgb = cv2.cvtColor(centre_square(image), cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 127.5 - 1.0)[np.newaxis]
