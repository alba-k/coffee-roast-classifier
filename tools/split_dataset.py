"""Split dataset/curated into dataset/split/{train,val,test} (70/20/10)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_dataset.splitter import SPLITS, build_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--block", type=int, default=5, help="consecutive photos kept together")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split folder")
    ap.add_argument("--resume", action="store_true", help="keep copying into an existing split folder (after an interruption)")
    a = ap.parse_args()

    out = a.dataset / "split"
    if out.exists() and any(out.iterdir()) and not (a.force or a.resume):
        print(f"{out} already exists. Use --resume to finish it or --force to rebuild it.")
        return 1
    if out.exists() and a.force and not a.resume:
        import shutil
        shutil.rmtree(out)
    counts = build_split(a.dataset / "curated", out, tuple(sorted(d.name for d in (a.dataset / "curated").iterdir() if d.is_dir())), (a.train, a.val, a.test), a.block, a.seed)
    print(f"{'class':<14}" + "".join(f"{s:>7}" for s in SPLITS))
    for label, row in counts.items():
        print(f"{label:<14}" + "".join(f"{row[s]:>7}" for s in SPLITS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
