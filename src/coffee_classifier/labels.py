"""Class names: read labels.txt and match them with the dataset folder names."""

from __future__ import annotations

from pathlib import Path

# Folder name in the dataset -> name used in Teachable Machine, when they differ.
FOLDER_ALIASES = {"nadaotro": "fondo"}


def load_labels(path: Path) -> tuple[str, ...]:
    """labels.txt lines look like '0 Verde'."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            names.append(line.split(" ", 1)[1] if line[0].isdigit() and " " in line else line)
    return tuple(names)


def _key(name: str) -> str:
    return name.lower().replace("_", "").replace(" ", "")


def folder_to_label(folder: str, labels: tuple[str, ...]) -> str | None:
    """'medio_oscuro' -> 'MedioOscuro', 'nada_otro' -> 'Fondo'. None if there is no match."""
    key = FOLDER_ALIASES.get(_key(folder), _key(folder))
    for label in labels:
        if _key(label) == key:
            return label
    return None
