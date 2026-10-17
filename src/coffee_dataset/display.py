"""Display implementation backed by an OpenCV window."""

from __future__ import annotations

import cv2

from .domain import Frame


class OpenCvWindow:
    def __init__(self, title: str = "COFFEE//DATASET") -> None:
        self._title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(title, 40, 40)
        try:  # keep the window above VS Code / the terminal so keys reach it
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    def show(self, frame: Frame) -> int:
        cv2.imshow(self._title, frame)
        key = cv2.waitKey(1)
        return -1 if key == -1 else key & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()
