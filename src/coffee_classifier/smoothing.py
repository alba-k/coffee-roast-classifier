"""Decorator: steadier answers on a live camera (average of the last few frames)."""

from __future__ import annotations

from collections import deque

from .domain import Classifier, Image, Prediction


class SmoothedClassifier:
    def __init__(self, inner: Classifier, window: int = 5) -> None:
        self.labels = inner.labels
        self._inner = inner
        self._recent: deque[dict[str, float]] = deque(maxlen=max(1, window))

    def predict(self, image: Image) -> Prediction:
        self._recent.append(self._inner.predict(image).scores)
        mean = {n: sum(s[n] for s in self._recent) / len(self._recent) for n in self.labels}
        best = max(mean, key=mean.get)
        return Prediction(best, mean[best], mean)
