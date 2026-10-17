"""Consistency checks between stages: the numbers must add up from one stage to the next."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    text: str
    ok: bool
    detail: str


def _t(results, key: str) -> int | None:
    r = results.get(key)
    return None if r is None else r.total


def run_checks(results: dict) -> list[Check]:
    g = lambda k: _t(results, k)
    out: list[Check] = []

    def eq(text, left, right):
        if left is None or right is None:
            out.append(Check(text, False, "folder missing"))
        else:
            out.append(Check(text, left == right, f"{left} vs {right}"))

    eq("first background crop + white crop = merged dataset (B2 + B4 = C1)",
       None if None in (g("B2"), g("B4")) else g("B2") + g("B4"), g("C1"))
    eq("model 2 split adds up to the merged dataset (C3 = C1)", g("C3"), g("C1"))
    eq("classifier project data = model 2 split (D2 = C3)", g("D2"), g("C3"))
    eq("model 1 split = first background crop (C2 = B2)", g("C2"), g("B2"))
    if results.get("C3") and results.get("D1"):
        train = sum(results["C3"].split["train"].values())
        d1 = results["D1"].total
        out.append(Check("uploaded to Teachable Machine = train photos minus exact duplicates (D1 <= C3 train)",
                         d1 <= train, f"{d1} uploaded, {train} in train, {train - d1} omitted"))
    return out
