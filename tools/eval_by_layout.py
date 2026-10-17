"""Accuracy split by how the beans were placed: grupo / disperso / individual (from the file name).

    python tools\eval_by_layout.py --split data\val
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_classifier.evaluate import evaluate_split  # noqa: E402
from coffee_classifier.labels import load_labels  # noqa: E402
from coffee_classifier.onnx_classifier import OnnxClassifier  # noqa: E402


def layout_of(name: str) -> str:
    for key in ("individual", "disperso", "grupo"):
        if f"_{key}_" in name:
            return key
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    a = ap.parse_args()
    labels = load_labels(a.model_dir / "labels.txt")
    result = evaluate_split(OnnxClassifier(a.model_dir / "model.onnx", labels), a.split)

    n: dict[tuple[str, str], int] = defaultdict(int)
    bad: dict[tuple[str, str], int] = defaultdict(int)
    for cls in (p for p in a.split.iterdir() if p.is_dir()):
        for f in cls.iterdir():
            if f.is_file():
                n[(cls.name, layout_of(f.name))] += 1
    for f, true, _p, _c in result.wrong:
        cls = Path(f).parent.name
        bad[(cls, layout_of(Path(f).name))] += 1

    layouts = ("grupo", "disperso", "individual")
    print(f"\n{'class':<14}" + "".join(f"{l:>16}" for l in layouts))
    tot = {l: [0, 0] for l in layouts}
    for cls in sorted({c for c, _ in n}):
        row = ""
        for l in layouts:
            k, w = n.get((cls, l), 0), bad.get((cls, l), 0)
            tot[l][0] += k - w; tot[l][1] += k
            row += f"{f'{k - w}/{k}':>16}"
        print(f"{cls:<14}{row}")
    print(f"{'ALL':<14}" + "".join(f"{f'{tot[l][0]}/{tot[l][1]}':>16}" for l in layouts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
