"""Trigger implementations: WHEN to take a photo."""

from __future__ import annotations


class IntervalTrigger:
    """Fires every `interval_s` seconds."""

    def __init__(self, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self._interval = interval_s
        self._next = 0.0

    def reset(self, now: float) -> None:
        self._next = now + self._interval

    def seconds_left(self, now: float) -> float:
        return max(0.0, self._next - now)

    def should_fire(self, now: float) -> bool:
        return now >= self._next

    def mark_fired(self, now: float) -> None:
        self._next = now + self._interval
