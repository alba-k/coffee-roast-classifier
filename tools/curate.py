"""Review raw photos one by one: K keep, D discard (hand, blur...), U undo, Q quit.

Kept photos are copied to dataset/curated/<label>/. Progress is saved, so you
can quit and continue later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from coffee_dataset.curation import DISCARD, KEEP, CurationLog  # noqa: E402

ACCENT = (247, 85, 168)
TEXT = (225, 215, 215)


def load(path: Path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = ap.parse_args()
    raw, curated = args.dataset / "raw", args.dataset / "curated"
    log = CurationLog(raw, curated)
    labels = tuple(sorted(d.name for d in raw.iterdir() if d.is_dir()))
    todo = log.pending(labels)
    i = 0
    while i < len(todo):
        rel = todo[i]
        img = load(raw / rel)
        kept, dropped = log.summary()
        cv2.putText(img, rel, (12, 28), cv2.FONT_HERSHEY_PLAIN, 1.4, ACCENT, 1, cv2.LINE_AA)
        cv2.putText(img, f"{i + 1}/{len(todo)}  kept {kept}  discarded {dropped}   [K]eep [D]iscard [U]ndo [Q]uit",
                    (12, img.shape[0] - 14), cv2.FONT_HERSHEY_PLAIN, 1.2, TEXT, 1, cv2.LINE_AA)
        cv2.imshow("COFFEE//CURATE", img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("k"):
            log.decide(rel, KEEP)
            i += 1
        elif key == ord("d"):
            log.decide(rel, DISCARD)
            i += 1
        elif key == ord("u"):
            undone = log.undo()
            if undone in todo:
                i = todo.index(undone)
    cv2.destroyAllWindows()
    kept, dropped = log.summary()
    print(f"kept {kept}, discarded {dropped}, pending {len(log.pending(labels))}")


if __name__ == "__main__":
    main()
