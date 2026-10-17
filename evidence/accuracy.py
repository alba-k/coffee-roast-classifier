"""Prove the accuracy figures: run each model on photos it never saw and print correct / total.

    python -m evidence.accuracy                 # both models, validation and test
    python -m evidence.accuracy --model 2       # only model 2
    python -m evidence.accuracy --errors        # also list every wrong photo

For each model it prints: the model file fingerprint (SHA-256), the photos checked, how many were
right, the accuracy with a 95 % confidence interval, the split by background (old paper vs white
sheet, from the date in the file name) and a leak check (no photo name shared between train and
val/test). Everything is recomputed from the photos; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .stages import PROJECT_ROOT, RECORDER_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coffee_classifier.evaluate import Evaluation, evaluate_split  # noqa: E402
from coffee_classifier.labels import load_labels  # noqa: E402
from coffee_classifier.onnx_classifier import OnnxClassifier  # noqa: E402

OLD_BACKGROUND_DAY = "20261015"      # photos of the first session; later days are the white sheet
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ModelSpec:
    number: int
    name: str
    model_dir: Path
    splits_dir: Path          # holds train / val / test


MODELS = (
    ModelSpec(1, "Model 1 (first background only)", RECORDER_ROOT / "model", RECORDER_ROOT / "dataset_final" / "split"),
    ModelSpec(2, "Model 2 (two backgrounds)", PROJECT_ROOT / "model", PROJECT_ROOT / "data"),
)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a proportion (Wilson score)."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def background_of(file_name: str) -> str:
    m = re.search(r"_(\d{8})_\d{6}_\d{3}", file_name)
    return "old" if m and m.group(1) == OLD_BACKGROUND_DAY else "white"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_names(folder: Path) -> set[str]:
    return {f.name for f in folder.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES}


def count_by_background(folder: Path) -> dict[str, int]:
    counts = {"old": 0, "white": 0}
    for name in image_names(folder):
        counts[background_of(name)] += 1
    return counts


def report_split(ev: Evaluation, split_dir: Path, split: str, show_errors: bool) -> None:
    total = ev.total
    correct = total - len(ev.wrong)
    lo, hi = wilson_interval(correct, total)
    print(f"  {split:<5} {correct:>4} of {total:<4} correct = {correct / total:6.1%}   95 % CI [{lo:.1%} , {hi:.1%}]")
    available = count_by_background(split_dir)
    wrong_by_bg = {"old": 0, "white": 0}
    for f, _t, _p, _c in ev.wrong:
        wrong_by_bg[background_of(Path(f).name)] += 1
    parts = []
    for bg in ("old", "white"):
        n = available[bg]
        if n:
            parts.append(f"{bg} background {n - wrong_by_bg[bg]}/{n} ({(n - wrong_by_bg[bg]) / n:.1%})")
    print(f"        by background: " + "   ".join(parts))
    if show_errors:
        for f, true, pred, conf in sorted(ev.wrong, key=lambda r: -r[3]):
            print(f"        wrong: {f}  real={true}  answer={pred}  confidence={conf:.3f}")


def leak_check(splits_dir: Path) -> None:
    train = image_names(splits_dir / "train")
    for split in ("val", "test"):
        shared = train & image_names(splits_dir / split)
        print(f"  leak check train/{split}: {len(shared)} shared photo names  ->  {'OK' if not shared else 'PROBLEM'}")


def run_model(spec: ModelSpec, show_errors: bool) -> None:
    print(f"\n{'=' * 72}\n{spec.name}\n{'=' * 72}")
    onnx = spec.model_dir / "model.onnx"
    if not onnx.is_file() or not (spec.splits_dir / "val").is_dir():
        print("  model or photo folders not found:", onnx, spec.splits_dir)
        return
    print(f"  model file : {onnx}")
    print(f"  SHA-256    : {sha256(onnx)}")
    labels = load_labels(spec.model_dir / "labels.txt")
    print(f"  classes    : {', '.join(labels)}")
    classifier = OnnxClassifier(onnx, labels)
    for split in ("val", "test"):
        report_split(evaluate_split(classifier, spec.splits_dir / split), spec.splits_dir / split, split, show_errors)
    leak_check(spec.splits_dir)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m evidence.accuracy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=int, choices=[m.number for m in MODELS], help="only this model")
    ap.add_argument("--errors", action="store_true", help="list every wrong photo")
    a = ap.parse_args()
    for spec in MODELS:
        if a.model in (None, spec.number):
            run_model(spec, a.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
