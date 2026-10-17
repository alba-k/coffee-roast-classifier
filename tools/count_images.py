"""Count the photos per class in each model's train / val / test folders.

    python tools\count_images.py
    python tools\count_images.py --model "Model 3=E:\\some\\folder"   (folder holds train, val, test)
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")

# Display name used in the report -> folder name on disk.
CLASS_ORDER = ("verde", "claro", "medio", "medio_oscuro", "oscuro", "quemado", "nada_otro")
DEFAULT_MODELS = {
    "Model 1 (first dataset, one background)": ROOT.parent / "coffee-dataset" / "dataset_final" / "split",
    "Model 2 (two backgrounds)": ROOT / "data",
}


def count_split(folder: Path) -> dict[str, int]:
    """{class name: number of images} for one split folder."""
    if not folder.is_dir():
        return {}
    return {
        c.name: sum(1 for f in c.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        for c in sorted(p for p in folder.iterdir() if p.is_dir())
    }


def print_model(name: str, root: Path) -> None:
    print(f"\n{'=' * 60}\n{name}\n{root}\n{'=' * 60}")
    counts = {s: count_split(root / s) for s in SPLITS}
    if not any(counts.values()):
        print("  (folder not found)")
        return
    classes = [c for c in CLASS_ORDER if any(c in counts[s] for s in SPLITS)]
    classes += sorted({c for s in SPLITS for c in counts[s]} - set(classes))
    print(f"{'class':<14}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'total':>8}")
    for c in classes:
        row = [counts[s].get(c, 0) for s in SPLITS]
        print(f"{c:<14}" + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    totals = [sum(counts[s].values()) for s in SPLITS]
    print("-" * 46)
    print(f"{'TOTAL':<14}" + "".join(f"{v:>8}" for v in totals) + f"{sum(totals):>8}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar='"NAME=FOLDER"')
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.model) if a.model else DEFAULT_MODELS
    for name, folder in models.items():
        print_model(name, Path(folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
