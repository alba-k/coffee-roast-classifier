"""Where the answers go. Add a serial / LED-ring / servo sink here without touching the rest."""

from __future__ import annotations

from .domain import Prediction, ResultSink


class ConsoleSink:
    """Prints only when the answer changes (no spam)."""

    def __init__(self) -> None:
        self._last = ""

    def emit(self, prediction: Prediction) -> None:
        if prediction.label != self._last:
            self._last = prediction.label
            print(f"[ ok ] {prediction.label:<12} {prediction.confidence:.0%}")


class CompositeSink:
    def __init__(self, sinks: list[ResultSink]) -> None:
        self._sinks = sinks

    def emit(self, prediction: Prediction) -> None:
        for sink in self._sinks:
            sink.emit(prediction)
