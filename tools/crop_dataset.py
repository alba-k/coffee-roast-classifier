"""Crop the curated photos to the well-lit centre (no dark edges) into a new, final dataset folder.

    python tools\\crop_dataset.py --dataset dataset_closed --out dataset_final
    python tools\\split_dataset.py --dataset dataset_final

Nothing in --dataset is modified. Photos that lose all their beans in the crop are listed in
<out>/crop_dropped.csv and left out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_dataset.batchcrop import crop_curated, write_dropped  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("dataset_closed"))
    ap.add_argument("--out", type=Path, default=Path("dataset_final"))
    a = ap.parse_args()

    curated = a.dataset / "curated"
    if not curated.is_dir():
        print(f"No {curated} folder: run tools\\auto_curate.py first.")
        return 1
    labels = tuple(sorted(d.name for d in curated.iterdir() if d.is_dir()))
    report = crop_curated(curated, a.out, labels)
    a.out.mkdir(parents=True, exist_ok=True)
    write_dropped(report, a.out / "crop_dropped.csv")

    print(f"{'class':<14}{'cropped':>8}")
    for label, n in report.written.items():
        print(f"{label:<14}{n:>8}")
    print(f"\nDropped (no beans left in the crop): {len(report.dropped)}  -> {a.out / 'crop_dropped.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
