"""Notifier implementations: how the person is told what happens.

Add a TelegramNotifier here later and plug it into CompositeNotifier.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence

from .domain import Notifier

try:  # Windows only
    import winsound
except ImportError:  # pragma: no cover
    winsound = None  # type: ignore[assignment]


class ConsoleNotifier:
    def info(self, message: str) -> None:
        guide_line = message == "" or message.startswith(("=", " ", "STEP", "Click", "A beep", "Changing"))
        print(message if guide_line else f"[ .. ] {message}")

    def warn(self, label: str) -> None:
        pass  # the HUD already shows the countdown

    def saved(self, label: str, count: int, target: int) -> None:
        print(f"[ ok ] {label:<13} {count:>4}/{target}")


class BeepNotifier:
    """Short beep before the shot (hands out!) and a low beep when saved."""

    def info(self, message: str) -> None:
        pass

    def warn(self, label: str) -> None:
        self._beep(1200, 120)

    def saved(self, label: str, count: int, target: int) -> None:
        self._beep(600, 60)

    @staticmethod
    def _beep(freq: int, ms: int) -> None:
        """Beep in a thread: winsound.Beep blocks, and the window must not wait."""
        def play() -> None:
            if winsound is not None:
                winsound.Beep(freq, ms)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()

        threading.Thread(target=play, daemon=True).start()


class CompositeNotifier:
    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        self._notifiers = list(notifiers)

    def info(self, message: str) -> None:
        for n in self._notifiers:
            n.info(message)

    def warn(self, label: str) -> None:
        for n in self._notifiers:
            n.warn(label)

    def saved(self, label: str, count: int, target: int) -> None:
        for n in self._notifiers:
            n.saved(label, count, target)
