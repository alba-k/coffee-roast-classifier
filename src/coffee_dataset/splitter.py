"""Stage 3: curated -> split/{train,val,test}/<label>.

Photos taken seconds apart look almost identical. If they were shuffled one by
one, near-duplicates would land in train AND test and the accuracy would be
inflated. So consecutive photos are grouped in blocks and whole blocks are
assigned to a split.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")


def split_blocks(
    files: list[Path],
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    block: int = 5,
    seed: int = 42,
) -> dict[str, list[Path]]:
    files = sorted(files)  # names start with label + layout + timestamp
    blocks = [files[i : i + block] for i in range(0, len(files), block)]
    result: dict[str, list[Path]] = {s: [] for s in SPLITS}
    if len(blocks) < 3:
        result["train"] = files
        return result
    random.Random(seed).shuffle(blocks)
    n_test = max(1, round(len(blocks) * ratios[2]))
    n_val = max(1, round(len(blocks) * ratios[1]))
    result["test"] = [f for b in blocks[:n_test] for f in b]
    result["val"] = [f for b in blocks[n_test : n_test + n_val] for f in b]
    result["train"] = [f for b in blocks[n_test + n_val :] for f in b]
    return result


def build_split(
    curated_root: Path,
    split_root: Path,
    labels: tuple[str, ...],
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    block: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Copy curated photos into split folders. Returns counts[label][split]."""
    counts: dict[str, dict[str, int]] = {}
    for label in labels:
        files = list((Path(curated_root) / label).glob("*.jpg"))
        parts = split_blocks(files, ratios, block, seed)
        counts[label] = {}
        for name, items in parts.items():
            dest = Path(split_root) / name / label
            dest.mkdir(parents=True, exist_ok=True)
            for item in items:
                if not (dest / item.name).exists():      # lets an interrupted split be resumed
                    shutil.copy2(item, dest / item.name)
            counts[label][name] = len(items)
    return counts
