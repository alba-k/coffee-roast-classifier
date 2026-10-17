"""Automatic first pass: keep the good raw photos, reject hands / blur / empty frames / mixed roasts.

    python tools\\auto_curate.py --dataset dataset_closed --dry-run     (only writes the report)
    python tools\\auto_curate.py --dataset dataset_closed               (copies the kept photos to curated/)

Raw photos are never touched. Afterwards `tools\\curate.py --dataset ...` still lets you fix by hand
anything it got wrong (photos already decided here are skipped there).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_dataset.autocurate import default_rules, judge, measure  # noqa: E402
from coffee_dataset.config import DEFAULT_LABELS  # noqa: E402
from coffee_dataset.curation import DISCARD, KEEP, CurationLog  # noqa: E402

REPORT = "auto_report.csv"


def layout_of(label: str, name: str) -> str:
    rest = name[len(label) + 1:] if name.startswith(label + "_") else name
    return rest.split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--dry-run", action="store_true", help="only write the report, copy nothing")
    a = ap.parse_args()

    raw, curated = a.dataset / "raw", a.dataset / "curated"
    log = CurationLog(raw, curated)
    rules = default_rules()
    todo = log.pending(DEFAULT_LABELS)
    if not todo:
        print("Nothing to do: every raw photo already has a decision.")
        return 0

    rows, counts = [], Counter()
    for i, rel in enumerate(todo, 1):
        label, name = rel.split("/", 1)
        image = cv2.imread(str(raw / rel))
        if image is None:
            reason, m = "unreadable", None
        else:
            m = measure(image)
            reason = judge(m, label, layout_of(label, name), rules)
        decision = KEEP if reason is None else DISCARD
        counts[(label, reason or "keep")] += 1
        rows.append((rel, decision, reason or "", *(("%.4f" % v for v in vars(m).values()) if m else ())))
        if not a.dry_run:
            log.decide(rel, decision)
        if i % 200 == 0:
            print(f"  {i}/{len(todo)}")

    curated.mkdir(parents=True, exist_ok=True)
    report = curated / REPORT
    previous: dict[str, list[str]] = {}
    if report.exists():                          # keep rows from earlier (interrupted) runs
        with report.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            previous = {row[0]: row for row in reader if row}
    previous.update({str(r[0]): [str(v) for v in r] for r in rows})
    with report.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(("file", "decision", "reason", "coverage", "largest_blob", "hand_area", "glove_area", "sharpness", "dark_share", "brightness"))
        w.writerows(previous.values())

    print("\nclass          kept  rejected  (reasons)")
    for label in DEFAULT_LABELS:
        kept = counts[(label, "keep")]
        why = {r: n for (lb, r), n in counts.items() if lb == label and r != "keep"}
        if kept or why:
            print(f"{label:<14} {kept:>4}  {sum(why.values()):>8}  {why}")
    print(f"\nReport: {curated / REPORT}" + ("   (dry run: nothing copied)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
