"""Abstractions (ports) of the classifier. Any model, source or output can be swapped."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

Image = np.ndarray  # BGR, as OpenCV reads it


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


class Classifier(Protocol):
    labels: tuple[str, ...]

    def predict(self, image: Image) -> Prediction: ...


class ResultSink(Protocol):
    """Receives every prediction (console, LED ring, servo, serial port...)."""

    def emit(self, prediction: Prediction) -> None: ...
