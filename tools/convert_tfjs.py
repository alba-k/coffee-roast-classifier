"""Teachable Machine export (TensorFlow.js zip) -> model/model.onnx + model/labels.txt.

    pip install -r requirements-convert.txt      (use a separate venv, Python 3.11)
    python tools\convert_tfjs.py model\source\tm-my-image-model.zip
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> int:
    zip_path = Path(sys.argv[1])
    out = Path("model")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zipfile.ZipFile(zip_path).extractall(tmp / "tfjs")
        labels = json.loads((tmp / "tfjs" / "metadata.json").read_text(encoding="utf-8"))["labels"]
        run = lambda *c: subprocess.run(c, check=True)
        run("tensorflowjs_converter", "--input_format=tfjs_layers_model",
            "--output_format=keras_saved_model", str(tmp / "tfjs" / "model.json"), str(tmp / "keras"))
        run(sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(tmp / "keras"),
            "--opset", "13", "--output", str(out / "model.onnx"))
    # Class order comes from the export's metadata.json, NOT from the order in the UI.
    (out / "labels.txt").write_text("".join(f"{i} {n}\n" for i, n in enumerate(labels)), encoding="utf-8")
    print("labels:", labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
