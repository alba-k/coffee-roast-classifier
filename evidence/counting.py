"""Count image files. Class names are normalised so 'medio_oscuro', 'MedioOscuro' and 'Fondo'/'nada_otro' match."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
CLASS_ORDER = ("verde", "claro", "medio", "mediooscuro", "oscuro", "quemado", "nadaotro")
ALIASES = {"fondo": "nadaotro"}
LABELS = {"mediooscuro": "medio_oscuro", "nadaotro": "nada_otro (Fondo)"}


def class_key(name: str) -> str:
    key = name.lower().replace("_", "").replace(" ", "")
    return ALIASES.get(key, key)


def class_label(key: str) -> str:
    return LABELS.get(key, key)


def sort_classes(keys) -> list[str]:
    known = [k for k in CLASS_ORDER if k in keys]
    return known + sorted(set(keys) - set(known))


def count_flat(folder: Path) -> dict[str, int] | None:
    """{class key: images} for <folder>/<class>/*.jpg. None if the folder does not exist."""
    if not folder.is_dir():
        return None
    counts: dict[str, int] = {}
    for sub in folder.iterdir():
        if sub.is_dir():
            n = sum(1 for f in sub.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
            counts[class_key(sub.name)] = counts.get(class_key(sub.name), 0) + n
    return counts


def count_split(folder: Path) -> dict[str, dict[str, int]] | None:
    if not folder.is_dir():
        return None
    return {s: (count_flat(folder / s) or {}) for s in SPLITS}


def total(counts: dict[str, int] | None) -> int:
    return sum(counts.values()) if counts else 0
