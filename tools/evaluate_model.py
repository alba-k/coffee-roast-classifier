"""How good is the model on photos it never saw?

    python tools\evaluate_model.py --split dataset_final\split\val
    python tools\evaluate_model.py --split dataset_final\split\test

Prints accuracy per class and a confusion table (rows = real class, columns = answer).
The wrong answers are saved to <split>_errors.csv so you can look at them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_classifier.evaluate import evaluate_split  # noqa: E402
from coffee_classifier.labels import load_labels  # noqa: E402
from coffee_classifier.onnx_classifier import OnnxClassifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True, help="folder with one sub-folder per class")
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    ap.add_argument("--errors", type=Path, default=None, help="where to save the wrong answers (csv)")
    a = ap.parse_args()

    labels = load_labels(a.model_dir / "labels.txt")
    result = evaluate_split(OnnxClassifier(a.model_dir / "model.onnx", labels), a.split)

    short = [name[:6] for name in labels]
    head = "real / answer"
    print(f"\n{head:<14}" + "".join(f"{s:>8}" for s in short) + f"{'recall':>9}")
    for i, name in enumerate(labels):
        row = "".join(f"{v:>8}" for v in result.matrix[i])
        print(f"{name:<14}{row}{result.recall(i):>9.0%}")
    print(f"\nPhotos: {result.total}   Accuracy: {result.accuracy:.1%}   Wrong: {len(result.wrong)}")

    out = a.errors or a.split.parent / f"{a.split.name}_errors.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(("file", "real", "answer", "confidence"))
        w.writerows((f, t, p, f"{c:.3f}") for f, t, p, c in sorted(result.wrong, key=lambda r: -r[3]))
    print(f"Errors: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
