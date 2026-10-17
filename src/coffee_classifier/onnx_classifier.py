"""Run the exported model with OpenCV's own DNN module (no TensorFlow needed)."""

from __future__ import annotations

from pathlib import Path

import cv2

from .domain import Image, Prediction
from .preprocess import to_model_input


class OnnxClassifier:
    def __init__(self, model_path: Path, labels: tuple[str, ...]) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.labels = labels
        self._net = cv2.dnn.readNetFromONNX(str(model_path))

    def predict(self, image: Image) -> Prediction:
        self._net.setInput(to_model_input(image))
        scores = self._net.forward()[0]
        best = int(scores.argmax())
        return Prediction(
            label=self.labels[best],
            confidence=float(scores[best]),
            scores={name: float(s) for name, s in zip(self.labels, scores)},
        )
