"""Check that model.onnx reproduces the original Teachable Machine (TF.js) model.

Converts the export zip to a Keras SavedModel (temp folder), feeds N random
224x224x3 inputs (seed 42, range [-1, 1] like Teachable Machine) to both models
and prints the maximum absolute difference between their 7 output scores.

    pip install -r requirements-convert.txt onnxruntime      (convert venv)
    python tools\\verify_conversion.py model\\source\\tm-my-image-model.zip --n 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("--onnx", type=Path, default=Path("model/model.onnx"))
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    import onnxruntime as ort
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zipfile.ZipFile(a.zip_path).extractall(tmp / "tfjs")
        subprocess.run(["tensorflowjs_converter", "--input_format=tfjs_layers_model",
                        "--output_format=keras_saved_model", str(tmp / "tfjs" / "model.json"),
                        str(tmp / "keras")], check=True)
        keras = tf.keras.models.load_model(tmp / "keras")
        sess = ort.InferenceSession(str(a.onnx))
        name = sess.get_inputs()[0].name
        x = np.random.default_rng(42).uniform(-1, 1, (a.n, 224, 224, 3)).astype("float32")
        ref = keras.predict(x, verbose=0)
        out = sess.run(None, {name: x})[0]
    print(f"inputs: {a.n}  max |diff|: {np.abs(ref - out).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
