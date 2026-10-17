"""Capture session: orchestrates the pieces through their abstractions."""

from __future__ import annotations

import time
from collections.abc import Callable

from .config import ARRANGEMENT_HELP, ARRANGEMENT_KEYS, Settings
from .guide import build_guide, class_message
from .domain import Camera, DatasetSink, Display, Notifier, Trigger
from .hud import HudState, PurpleHud

KEY_QUIT = (ord("q"), 27)  # q or ESC
KEY_TOGGLE = ord(" ")
KEY_SHOT = ord("s")


class CaptureSession:
    def __init__(
        self,
        settings: Settings,
        camera: Camera,
        trigger: Trigger,
        notifier: Notifier,
        sink: DatasetSink,
        display: Display,
        hud: PurpleHud,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._s = settings
        self._camera = camera
        self._trigger = trigger
        self._notifier = notifier
        self._sink = sink
        self._display = display
        self._hud = hud
        self._clock = clock
        self._current = settings.labels.index(settings.start_label) if settings.start_label in settings.labels else 0
        self._arrangement = (
            settings.arrangements.index(settings.start_layout) if settings.start_layout in settings.arrangements else 0
        )
        self._running = False
        self._warned = False
        self._message = ""

    # ---- public -------------------------------------------------------
    def run(self) -> None:
        self._camera.open()
        for line in build_guide(self._s.labels, self._s.arrangements, self._s.interval_s, self._s.target_per_class):
            self._notifier.info(line)
        if self._s.autostart:
            self._notifier.info(
                f"Starting now: {class_message(self._current + 1, self._label)} | layout {self._tag}. SPACE = pause."
            )
            self._running = True
            self._trigger.reset(self._clock())
        try:
            while True:
                frame = self._camera.read()
                now = self._clock()
                if self._running:
                    self.tick(frame, now)
                shown = self._hud.render(frame, self._hud_state(now))
                if self.handle_key(self._display.show(shown), frame, now):
                    break
        finally:
            self._camera.close()
            self._display.close()

    def tick(self, frame, now: float) -> None:
        """Called every loop while recording."""
        label = self._label
        if self._sink.count(label) >= self._s.target_per_class:
            self._stop(f"{label}: target reached")
            return
        if not self._warned and self._trigger.seconds_left(now) <= self._s.warn_s:
            self._warned = True
            self._notifier.warn(label)
        if self._trigger.should_fire(now):
            self._shoot(frame, now)

    def handle_key(self, key: int, frame, now: float) -> bool:
        """Returns True when the app must quit."""
        if key in KEY_QUIT:
            return True
        if key == KEY_TOGGLE:
            self._running = not self._running
            if self._running:
                self._trigger.reset(now)
                self._warned = False
                self._message = ""
            return False
        if key == KEY_SHOT:
            self._shoot(frame, now)
            return False
        if 0 <= key < 256 and chr(key).lower() in ARRANGEMENT_KEYS:
            self._arrangement = ARRANGEMENT_KEYS.index(chr(key).lower())
            self._message = f"layout: {self._tag}"
            self._notifier.info(f"Layout = {self._tag}: {ARRANGEMENT_HELP.get(self._tag, '')}")
            return False
        index = key - ord("1")
        if 0 <= index < len(self._s.labels):
            self._current = index
            self._running = False  # pause so you can change the beans
            self._message = f"class: {self._label}"
            self._notifier.info(class_message(index + 1, self._label) + "  [paused: press SPACE]")
        return False

    # ---- internals ----------------------------------------------------
    @property
    def _label(self) -> str:
        return self._s.labels[self._current]

    @property
    def _tag(self) -> str:
        return self._s.arrangements[self._arrangement]

    def _shoot(self, frame, now: float) -> None:
        self._sink.save(frame, self._label, self._tag)
        self._trigger.mark_fired(now)
        self._warned = False
        self._notifier.saved(self._label, self._sink.count(self._label), self._s.target_per_class)

    def _stop(self, message: str) -> None:
        self._running = False
        self._message = message
        self._notifier.info(message)

    def _hud_state(self, now: float) -> HudState:
        return HudState(
            labels=self._s.labels,
            counts=tuple(self._sink.count(lb) for lb in self._s.labels),
            current=self._current,
            target=self._s.target_per_class,
            running=self._running,
            seconds_left=self._trigger.seconds_left(now),
            interval_s=self._s.interval_s,
            arrangement=self._tag,
            message=self._message,
        )
