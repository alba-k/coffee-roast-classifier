"""Session logic tested with fakes (no camera, no window, no real time)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_dataset.config import Settings  # noqa: E402
from coffee_dataset.hud import PurpleHud  # noqa: E402
from coffee_dataset.session import CaptureSession  # noqa: E402
from coffee_dataset.storage import FolderDatasetStore  # noqa: E402
from coffee_dataset.triggers import IntervalTrigger  # noqa: E402


class FakeNotifier:
    def __init__(self):
        self.events = []

    def info(self, message):
        self.events.append(("info", message))

    def warn(self, label):
        self.events.append(("warn", label))

    def saved(self, label, count, target):
        self.events.append(("saved", label, count))


def make(tmp_path, target=3, interval=3.0):
    s = Settings(output_dir=tmp_path, target_per_class=target, interval_s=interval, warn_s=1.0)
    notifier = FakeNotifier()
    session = CaptureSession(
        s, camera=None, trigger=IntervalTrigger(interval), notifier=notifier,
        sink=FolderDatasetStore(tmp_path, s.labels), display=None, hud=PurpleHud(),
    )
    return session, notifier, s


FRAME = np.full((120, 160, 3), 200, dtype=np.uint8)


def test_records_on_interval_and_warns(tmp_path):
    session, notifier, _ = make(tmp_path)
    session.handle_key(ord("2"), FRAME, 0.0)       # class 'claro'
    session.handle_key(ord(" "), FRAME, 0.0)       # start
    for t in (0.5, 1.9, 2.1, 3.0, 3.5, 6.0):
        session.tick(FRAME, t)
    saved = [e for e in notifier.events if e[0] == "saved"]
    assert [e[2] for e in saved] == [1, 2]          # shots at t=3.0 and t=6.0
    assert ("warn", "claro") in notifier.events
    assert (tmp_path / "claro").is_dir()
    assert len(list((tmp_path / "claro").glob("*.jpg"))) == 2


def test_stops_at_target(tmp_path):
    session, notifier, _ = make(tmp_path, target=2)
    session.handle_key(ord("1"), FRAME, 0.0)
    session.handle_key(ord(" "), FRAME, 0.0)
    for t in range(0, 40):
        session.tick(FRAME, float(t))
    assert session._sink.count("verde") == 2
    assert session._running is False


def test_switching_class_pauses(tmp_path):
    session, _, _ = make(tmp_path)
    session.handle_key(ord(" "), FRAME, 0.0)
    assert session._running is True
    session.handle_key(ord("5"), FRAME, 1.0)
    assert session._running is False
    assert session._label == "oscuro"


def test_manifest_and_hud(tmp_path):
    session, _, s = make(tmp_path)
    session.handle_key(ord("s"), FRAME, 0.0)       # manual shot
    rows = (tmp_path / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and rows[1].startswith("verde/verde_")
    out = PurpleHud().render(FRAME, session._hud_state(0.0))
    assert out.shape[0] == FRAME.shape[0] and out.shape[1] > FRAME.shape[1]
    assert np.array_equal(out[:, -FRAME.shape[1]:], FRAME)   # image part is intact
    assert np.array_equal(FRAME, np.full((120, 160, 3), 200, dtype=np.uint8))  # original untouched


def test_quit_keys(tmp_path):
    session, _, _ = make(tmp_path)
    assert session.handle_key(ord("q"), FRAME, 0.0) is True
    assert session.handle_key(27, FRAME, 0.0) is True
    assert session.handle_key(-1, FRAME, 0.0) is False


def test_test_mode_settings(monkeypatch):
    import runpy
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    monkeypatch.setattr(sys, "argv", ["main.py", "--test", "--camera", "1"])
    ns = runpy.run_path(str(Path(__file__).resolve().parent.parent / "main.py"), run_name="not_main")
    s = ns["parse_args"]()
    assert s.labels == ("prueba",) and s.target_per_class == 12 and s.output_dir == Path("dataset_test")


def test_preselected_class_layout_and_autostart(tmp_path):
    s = Settings(output_dir=tmp_path, start_label="verde", start_layout="grupo", autostart=True, interval_s=3.0)

    class OneFrameCamera:
        def __init__(self): self.n = 0
        def open(self): pass
        def read(self):
            self.n += 1
            return FRAME
        def close(self): pass

    class QuitAfter:
        def __init__(self): self.calls = 0
        def show(self, frame):
            self.calls += 1
            return ord("q") if self.calls >= 3 else -1
        def close(self): pass

    notifier = FakeNotifier()
    ticks = iter([0.0, 0.1, 0.2, 0.3])
    session = CaptureSession(
        s, camera=OneFrameCamera(), trigger=IntervalTrigger(3.0), notifier=notifier,
        sink=FolderDatasetStore(tmp_path, s.labels), display=QuitAfter(), hud=PurpleHud(),
        clock=lambda: next(ticks),
    )
    session.run()
    assert session._label == "verde" and session._tag == "grupo" and session._running is True
    assert any("Starting now" in e[1] for e in notifier.events if e[0] == "info")
