"""Measure the model on photos it never saw (split/val and split/test)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .domain import Classifier
from .labels import folder_to_label


@dataclass
class Evaluation:
    labels: tuple[str, ...]
    matrix: np.ndarray                      # rows = true class, columns = predicted class
    wrong: list[tuple[str, str, str, float]] = field(default_factory=list)  # file, true, predicted, confidence

    @property
    def total(self) -> int:
        return int(self.matrix.sum())

    @property
    def accuracy(self) -> float:
        return float(np.trace(self.matrix) / self.total) if self.total else 0.0

    def recall(self, index: int) -> float:
        row = self.matrix[index].sum()
        return float(self.matrix[index, index] / row) if row else float("nan")


def evaluate_split(classifier: Classifier, split_dir: Path) -> Evaluation:
    labels = classifier.labels
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    wrong: list[tuple[str, str, str, float]] = []
    for folder in sorted(d for d in Path(split_dir).iterdir() if d.is_dir()):
        true = folder_to_label(folder.name, labels)
        if true is None:
            continue
        for path in sorted(folder.glob("*.jpg")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            pred = classifier.predict(image)
            matrix[index[true], index[pred.label]] += 1
            if pred.label != true:
                wrong.append((f"{folder.name}/{path.name}", true, pred.label, pred.confidence))
    return Evaluation(labels, matrix, wrong)
