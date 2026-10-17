"""Plain-language help printed when the program starts."""

from __future__ import annotations

from .config import ARRANGEMENT_HELP, ARRANGEMENT_KEYS, CLASS_HELP


def build_guide(
    labels: tuple[str, ...],
    arrangements: tuple[str, ...],
    interval_s: float,
    target: int,
) -> list[str]:
    minutes = interval_s * target / 60
    lines = [
        "",
        "==== COFFEE//DATASET - HOW TO USE ====",
        "Click on the camera window first, then press keys there.",
        "",
        "STEP 1  pick the CLASS (what is in the box):",
    ]
    for number, label in enumerate(labels, start=1):
        lines.append(f"   press {number}  ->  {label:<13} {CLASS_HELP.get(label, '')}")
    lines += ["", "STEP 2  pick the LAYOUT (how the beans are placed):"]
    for key, name in zip(ARRANGEMENT_KEYS, arrangements):
        lines.append(f"   press {key.upper()}  ->  {name:<10} {ARRANGEMENT_HELP.get(name, '')}")
    lines += [
        "",
        "STEP 3  record:",
        f"   press SPACE  ->  start / pause (1 photo every {interval_s:g} s, {target} photos = {minutes:g} min)",
        "   press S      ->  take one photo right now",
        "   press Q      ->  quit",
        "",
        "A beep sounds just before each photo: move the beans, take your hand out!",
        "Changing the class pauses the recording; press SPACE again to continue.",
        "=" * 39,
        "",
    ]
    return lines


def class_message(number: int, label: str) -> str:
    return f"Class {number} = {label}: {CLASS_HELP.get(label, '')}"
