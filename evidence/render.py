"""Turn the counts into a readable text report and a CSV file."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .counting import SPLITS, class_label, sort_classes
from .stages import Stage


@dataclass(frozen=True)
class StageResult:
    stage: Stage
    flat: dict[str, int] | None = None
    split: dict[str, dict[str, int]] | None = None

    @property
    def exists(self) -> bool:
        return self.flat is not None or self.split is not None

    @property
    def total(self) -> int:
        if self.split is not None:
            return sum(sum(c.values()) for c in self.split.values())
        return sum(self.flat.values()) if self.flat else 0


def stage_text(r: StageResult) -> str:
    s = r.stage
    head = [f"[{s.key}] {s.title}", f"     {s.folder}"]
    if s.note:
        head.append(f"     ({s.note})")
    if not r.exists:
        return "\n".join(head + ["     folder not found", ""])
    lines = head
    if r.split is not None:
        keys = sort_classes({k for c in r.split.values() for k in c})
        lines.append(f"     {'class':<20}" + "".join(f"{sp:>8}" for sp in SPLITS) + f"{'total':>8}")
        for k in keys:
            row = [r.split[sp].get(k, 0) for sp in SPLITS]
            lines.append(f"     {class_label(k):<20}" + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
        sums = [sum(r.split[sp].values()) for sp in SPLITS]
        lines.append("     " + "-" * 44)
        lines.append(f"     {'TOTAL':<20}" + "".join(f"{v:>8}" for v in sums) + f"{sum(sums):>8}")
    else:
        lines.append(f"     {'class':<20}{'photos':>8}")
        for k in sort_classes(r.flat):
            lines.append(f"     {class_label(k):<20}{r.flat[k]:>8}")
        lines.append("     " + "-" * 28)
        lines.append(f"     {'TOTAL':<20}{r.total:>8}")
    return "\n".join(lines + [""])


def summary_text(results: list[StageResult]) -> str:
    lines = ["SUMMARY (photos per stage)", f"{'stage':<6}{'photos':>8}  description"]
    phase = ""
    for r in results:
        if r.stage.phase != phase:
            phase = r.stage.phase
            lines.append(f"-- {phase}")
        count = f"{r.total:>8}" if r.exists else f"{'n/a':>8}"
        lines.append(f"{r.stage.key:<6}{count}  {r.stage.title}")
    return "\n".join(lines)


def write_csv(results: list[StageResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(("stage", "title", "folder", "split", "class", "photos"))
        for r in results:
            if r.split is not None:
                for sp in SPLITS:
                    for k in sort_classes(r.split[sp]):
                        w.writerow((r.stage.key, r.stage.title, r.stage.folder, sp, class_label(k), r.split[sp][k]))
            elif r.flat is not None:
                for k in sort_classes(r.flat):
                    w.writerow((r.stage.key, r.stage.title, r.stage.folder, "", class_label(k), r.flat[k]))
