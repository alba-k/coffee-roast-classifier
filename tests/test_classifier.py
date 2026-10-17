"""Classifier pieces (no camera, no model file needed)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_classifier.domain import Prediction  # noqa: E402
from coffee_classifier.labels import folder_to_label, load_labels  # noqa: E402
from coffee_classifier.overlay import PurpleOverlay  # noqa: E402
from coffee_classifier.preprocess import centre_square, match_training_frame, to_model_input  # noqa: E402
from coffee_classifier.smoothing import SmoothedClassifier  # noqa: E402

LABELS = ("Verde", "Claro", "Medio", "MedioOscuro", "Oscuro", "Quemado", "Fondo")


class FakeModel:
    labels = LABELS

    def __init__(self, answers):
        self._answers = iter(answers)

    def predict(self, image):
        best = next(self._answers)
        scores = {n: (0.9 if n == best else 0.1 / 6) for n in LABELS}
        return Prediction(best, scores[best], scores)


def test_load_labels(tmp_path):
    f = tmp_path / "labels.txt"
    f.write_text("0 Verde\n1 Claro\n2 MedioOscuro\n", encoding="utf-8")
    assert load_labels(f) == ("Verde", "Claro", "MedioOscuro")


def test_folder_names_map_to_labels():
    assert folder_to_label("medio_oscuro", LABELS) == "MedioOscuro"
    assert folder_to_label("nada_otro", LABELS) == "Fondo"
    assert folder_to_label("defectuoso", LABELS) is None


def test_live_frame_is_cut_like_the_training_photos():
    frame = np.zeros((720, 1280, 3), np.uint8)
    seen = match_training_frame(frame)
    assert seen.shape[:2] == (450, 800)
    assert centre_square(seen).shape[:2] == (450, 450)


def test_model_input_shape_and_range():
    x = to_model_input(np.full((450, 800, 3), 255, np.uint8))
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert x.max() <= 1.0 and x.min() >= -1.0


def test_smoothing_ignores_a_single_odd_frame():
    model = SmoothedClassifier(FakeModel(["Oscuro", "Oscuro", "Quemado", "Oscuro"]), window=4)
    answers = [model.predict(None).label for _ in range(4)]
    assert answers[-1] == "Oscuro"


def test_overlay_adds_a_panel():
    frame = np.zeros((450, 800, 3), np.uint8)
    pred = FakeModel(["Verde"]).predict(None)
    out = PurpleOverlay().render(frame, pred, "Verde", (175, 0, 625, 450))
    assert out.shape[0] == 450 and out.shape[1] > 800
