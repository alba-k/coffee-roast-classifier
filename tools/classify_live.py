"""Tell the roast level of the beans in front of the camera, live.

    python tools\classify_live.py
    python tools\classify_live.py --camera 1 --smooth 8 --min-conf 0.7

Q quits. S saves what the model sees (to live_captures/) so a wrong answer can be studied.
The purple square is exactly the area the model looks at.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_classifier.domain import Prediction  # noqa: E402
from coffee_classifier.labels import load_labels  # noqa: E402
from coffee_classifier.onnx_classifier import OnnxClassifier  # noqa: E402
from coffee_classifier.overlay import PurpleOverlay  # noqa: E402
from coffee_classifier.preprocess import match_training_frame  # noqa: E402
from coffee_classifier.sinks import ConsoleSink  # noqa: E402
from coffee_classifier.smoothing import SmoothedClassifier  # noqa: E402
from coffee_dataset.config import Settings  # noqa: E402
from coffee_dataset.display import OpenCvWindow  # noqa: E402
from coffee_dataset.domain import CameraError  # noqa: E402
from coffee_dataset.factory import build_camera  # noqa: E402

UNSURE = "no estoy seguro"


def view_area(frame_shape) -> tuple[int, int, int, int]:
    """The model's centred square, in the coordinates of the frame being shown."""
    h, w = frame_shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return x0, y0, x0 + side, y0 + side


def save_capture(seen, prediction: Prediction, folder: Path = Path("live_captures")) -> None:
    """Keep the exact image the model saw, with its answer in the name, and print its average colour."""
    folder.mkdir(exist_ok=True)
    name = f"{prediction.label}_{prediction.confidence:.2f}_{time.strftime('%H%M%S')}.jpg"
    cv2.imwrite(str(folder / name), seen)
    h, w = seen.shape[:2]
    side = min(h, w)
    centre = seen[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
    r, g, b = np.asarray(centre).reshape(-1, 3).mean(0)[::-1]
    print(f"[ ok ] saved {folder / name}   mean RGB {r:.0f} {g:.0f} {b:.0f}")


def main() -> int:
    d = Settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=d.camera_index)
    ap.add_argument("--width", type=int, default=d.width)
    ap.add_argument("--height", type=int, default=d.height)
    ap.add_argument("--fps", type=float, default=d.fps)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    ap.add_argument("--smooth", type=int, default=5, help="average the last N frames (1 = off)")
    ap.add_argument("--min-conf", type=float, default=0.6, help="below this the answer is 'unsure'")
    a = ap.parse_args()

    settings = Settings(camera_index=a.camera, width=a.width, height=a.height, fps=a.fps,
                        mirror=a.mirror, use_crop=False)   # the dataset was captured without crop.json
    labels = load_labels(a.model_dir / "labels.txt")
    classifier = SmoothedClassifier(OnnxClassifier(a.model_dir / "model.onnx", labels), a.smooth)
    sink, overlay = ConsoleSink(), PurpleOverlay()

    camera = build_camera(settings)
    window = None
    try:
        camera.open()
        window = OpenCvWindow("COFFEE//CLASSIFIER")
        while True:
            frame = camera.read()
            seen = match_training_frame(frame)               # same size the training photos had
            prediction: Prediction = classifier.predict(seen)
            shown = prediction.label if prediction.confidence >= a.min_conf else UNSURE
            sink.emit(Prediction(shown, prediction.confidence, prediction.scores))
            key = window.show(overlay.render(seen, prediction, shown, view_area(seen.shape)))
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                save_capture(seen, prediction)
    except CameraError as exc:
        print(f"[fail] {exc}")
        return 1
    finally:
        camera.close()
        if window is not None:
            window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
