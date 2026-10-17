"""Stage 2: raw -> curated. Keeps a resumable log of keep/discard decisions."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

LOG_NAME = "decisions.csv"
KEEP, DISCARD = "keep", "discard"


class CurationLog:
    """dataset/curated/decisions.csv  (relative file, decision)."""

    def __init__(self, raw_root: Path, curated_root: Path) -> None:
        self._raw = Path(raw_root)
        self._curated = Path(curated_root)
        self._curated.mkdir(parents=True, exist_ok=True)
        self._log = self._curated / LOG_NAME
        self._decisions: dict[str, str] = {}
        self._order: list[str] = []
        if self._log.exists():
            with self._log.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self._decisions[row["file"]] = row["decision"]
                    self._order.append(row["file"])

    def pending(self, labels: tuple[str, ...]) -> list[str]:
        """Raw files (relative, 'label/name.jpg') without a decision yet, in time order."""
        files: list[str] = []
        for label in labels:
            folder = self._raw / label
            if folder.exists():
                files += [f"{label}/{p.name}" for p in sorted(folder.glob("*.jpg"))]
        return [f for f in files if f not in self._decisions]

    def decide(self, rel: str, decision: str) -> None:
        if decision == KEEP:
            dest = self._curated / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._raw / rel, dest)
        if rel not in self._decisions:
            self._order.append(rel)
        self._decisions[rel] = decision
        self._save()

    def undo(self) -> str | None:
        """Forget the last decision (and remove the curated copy). Returns the file."""
        if not self._order:
            return None
        rel = self._order.pop()
        if self._decisions.pop(rel, None) == KEEP:
            (self._curated / rel).unlink(missing_ok=True)
        self._save()
        return rel

    def summary(self) -> tuple[int, int]:
        values = list(self._decisions.values())
        return values.count(KEEP), values.count(DISCARD)

    def _save(self) -> None:
        with self._log.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(("file", "decision"))
            for rel in self._order:
                writer.writerow((rel, self._decisions[rel]))
