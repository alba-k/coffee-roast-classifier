"""Command line entry point: count the photos at every stage and check the numbers add up."""

from __future__ import annotations

import argparse
from pathlib import Path

from .checks import run_checks
from .counting import count_flat, count_split
from .render import StageResult, stage_text, summary_text, write_csv
from .stages import PROJECT_ROOT, STAGES


def collect(keys: set[str] | None) -> list[StageResult]:
    results = []
    for s in STAGES:
        if keys and s.key.upper() not in keys:
            continue
        if s.kind == "split":
            results.append(StageResult(s, split=count_split(s.folder)))
        else:
            results.append(StageResult(s, flat=count_flat(s.folder)))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m evidence", description=__doc__)
    ap.add_argument("--stage", nargs="*", help="only these stages, e.g. --stage A1 C3 D2")
    ap.add_argument("--summary", action="store_true", help="only the summary and the checks")
    ap.add_argument("--csv", type=Path, default=PROJECT_ROOT / "results" / "pipeline_counts.csv",
                    help="where to save the counts (default: results\\pipeline_counts.csv)")
    a = ap.parse_args()

    results = collect({k.upper() for k in a.stage} if a.stage else None)
    if not a.summary:
        phase = ""
        for r in results:
            if r.stage.phase != phase:
                phase = r.stage.phase
                print(f"\n{'=' * 64}\nPHASE {phase}\n{'=' * 64}\n")
            print(stage_text(r))
    print(summary_text(results))

    if not a.stage:
        print("\nCHECKS (do the numbers add up?)")
        for c in run_checks({r.stage.key: r for r in results}):
            print(f"  [{'OK' if c.ok else 'DIFF'}] {c.text}  ->  {c.detail}")
    write_csv(results, a.csv)
    print(f"\nSaved: {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
