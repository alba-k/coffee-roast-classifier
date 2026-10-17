"""Accuracy split by background: old paper (taken 2026-10-15) vs white sheet (2026-10-16/17).

    python tools\eval_by_background.py --split data\val
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_classifier.evaluate import evaluate_split  # noqa: E402
from coffee_classifier.labels import load_labels  # noqa: E402
from coffee_classifier.onnx_classifier import OnnxClassifier  # noqa: E402

OLD_DAY = "20261015"


def background_of(name: str) -> str:
    m = re.search(r"_(\d{8})_\d{6}_\d{3}", name)
    return "old" if m and m.group(1) == OLD_DAY else "white"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    a = ap.parse_args()

    labels = load_labels(a.model_dir / "labels.txt")
    result = evaluate_split(OnnxClassifier(a.model_dir / "model.onnx", labels), a.split)

    total: dict[tuple[str, str], int] = defaultdict(int)
    for cls in sorted(p for p in a.split.iterdir() if p.is_dir()):
        for f in cls.iterdir():
            if f.is_file():
                total[(cls.name, background_of(f.name))] += 1
    wrong: dict[tuple[str, str], int] = defaultdict(int)
    for f, true, _pred, _conf in result.wrong:
        wrong[(true, background_of(Path(f).name))] += 1

    print(f"\n{'class':<14}{'old ok/n':>12}{'white ok/n':>14}")
    sums = {"old": [0, 0], "white": [0, 0]}
    for cls in sorted({c for c, _ in total}):
        cells = []
        for bg in ("old", "white"):
            n = total.get((cls, bg), 0)
            w = sum(v for (t, b), v in wrong.items() if b == bg and t.lower().replace("_", "") == cls.lower().replace("_", ""))
            sums[bg][0] += n - w
            sums[bg][1] += n
            cells.append(f"{n - w}/{n}")
        print(f"{cls:<14}{cells[0]:>12}{cells[1]:>14}")
    for bg in ("old", "white"):
        ok, n = sums[bg]
        print(f"{bg:<6} accuracy: {ok}/{n} = {ok / n:.1%}" if n else f"{bg}: no photos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
