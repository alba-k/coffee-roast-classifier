"""Curation log and split."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coffee_dataset.curation import DISCARD, KEEP, CurationLog  # noqa: E402
from coffee_dataset.splitter import build_split, split_blocks  # noqa: E402
from coffee_dataset.storage import FolderDatasetStore  # noqa: E402

BIG = np.full((720, 1280, 3), 120, dtype=np.uint8)


def test_saved_photo_is_downscaled_and_tagged(tmp_path):
    store = FolderDatasetStore(tmp_path, ("claro",), jpeg_quality=85, max_width=640)
    path = store.save(BIG, "claro", "grupo")
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (360, 640)
    assert path.name.startswith("claro_grupo_")
    assert "grupo" in (tmp_path / "manifest.csv").read_text()


def make_raw(root: Path, label: str, n: int) -> None:
    (root / label).mkdir(parents=True)
    for i in range(n):
        (root / label / f"{label}_grupo_20261015_10{i // 60:02d}{i % 60:02d}_000.jpg").write_bytes(b"x")


def test_curation_resume_and_undo(tmp_path):
    raw, cur = tmp_path / "raw", tmp_path / "curated"
    make_raw(raw, "claro", 3)
    log = CurationLog(raw, cur)
    todo = log.pending(("claro",))
    assert len(todo) == 3
    log.decide(todo[0], KEEP)
    log.decide(todo[1], DISCARD)
    assert (cur / todo[0]).exists() and not (cur / todo[1]).exists()
    again = CurationLog(raw, cur)                      # resumes from disk
    assert again.pending(("claro",)) == [todo[2]]
    assert again.undo() == todo[1]
    assert again.undo() == todo[0] and not (cur / todo[0]).exists()
    assert len(again.pending(("claro",))) == 3


def test_split_has_no_overlap_and_keeps_blocks(tmp_path):
    files = [Path(f"claro_{i:03d}.jpg") for i in range(100)]
    parts = split_blocks(files, block=5)
    assert sum(len(v) for v in parts.values()) == 100
    all_names = [f for v in parts.values() for f in v]
    assert len(set(all_names)) == 100
    assert len(parts["test"]) == 10 and len(parts["val"]) == 20 and len(parts["train"]) == 70
    idx = sorted(int(f.stem[-3:]) // 5 for f in parts["test"])
    assert all(idx.count(b) == 5 for b in set(idx))    # whole blocks only


def test_build_split_copies_files(tmp_path):
    cur = tmp_path / "curated"
    make_raw(cur, "claro", 40)
    counts = build_split(cur, tmp_path / "split", ("claro",))
    assert sum(counts["claro"].values()) == 40
    assert len(list((tmp_path / "split" / "test" / "claro").glob("*.jpg"))) == counts["claro"]["test"]


def test_crop_square_and_camera():
    from coffee_dataset.crop import CroppedCamera, CropRegion

    region = CropRegion.square_from_pixels(200, 100, 500, 400, 1280, 720)   # 500x400 -> 400x400
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = region.apply(frame)
    assert out.shape[:2] == (400, 400)

    class Fake:
        def open(self): pass
        def read(self): return frame
        def close(self): pass

    assert CroppedCamera(Fake(), region).read().shape[:2] == (400, 400)


def test_crop_file_roundtrip(tmp_path):
    from coffee_dataset.crop import CropRegion

    path = tmp_path / "crop.json"
    assert CropRegion.load(path) is None
    CropRegion(0.1, 0.2, 0.3, 0.4).save(path)
    assert CropRegion.load(path) == CropRegion(0.1, 0.2, 0.3, 0.4)


def test_guide_lists_every_class_and_key():
    from coffee_dataset.config import ARRANGEMENTS, DEFAULT_LABELS
    from coffee_dataset.guide import build_guide

    text = "\n".join(build_guide(DEFAULT_LABELS, ARRANGEMENTS, 5.0, 60))
    for label in DEFAULT_LABELS:
        assert label in text
    for needle in ("press 1", "press 8", "press G", "press D", "press I", "SPACE", "5 min"):
        assert needle in text


def test_background_camera_serves_frames_and_reports_stall():
    import time

    from coffee_dataset.cameras import BackgroundCamera
    from coffee_dataset.domain import CameraError

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    class Fast:
        def open(self): pass
        def read(self): return frame
        def close(self): pass

    cam = BackgroundCamera(Fast(), fps=50)
    cam.open()
    try:
        assert cam.read().shape == (10, 10, 3)
    finally:
        cam.close()

    class Frozen:                      # a driver that hangs after the first frame
        n = 0
        def open(self): pass
        def read(self):
            Frozen.n += 1
            if Frozen.n > 1:
                time.sleep(5)
            return frame
        def close(self): pass

    cam = BackgroundCamera(Frozen(), fps=50, stale_after_s=0.3)
    cam.open()
    try:
        started = time.monotonic()
        cam.read()                       # first frame is fine
        time.sleep(0.5)
        try:
            cam.read()
            raised = False
        except CameraError:
            raised = True
        assert raised
        assert time.monotonic() - started < 2.0     # the main loop was never stuck for seconds
    finally:
        cam.close()


def test_camera_controls_apply_and_roundtrip(tmp_path):
    from coffee_dataset.controls import CameraControls

    class FakeCap:
        def __init__(self): self.calls = {}
        def set(self, prop, value): self.calls[prop] = value; return True

    c = CameraControls(auto_wb=False, wb_temperature=3800, auto_exposure=False, exposure=-7.0)
    cap = FakeCap()
    c.apply_to(cap)
    assert cap.calls[cv2.CAP_PROP_AUTO_WB] == 0 and cap.calls[cv2.CAP_PROP_WB_TEMPERATURE] == 3800
    assert cap.calls[cv2.CAP_PROP_AUTO_EXPOSURE] == 0.25 and cap.calls[cv2.CAP_PROP_EXPOSURE] == -7.0

    auto = FakeCap()
    CameraControls().apply_to(auto)                      # defaults leave auto modes on
    assert auto.calls[cv2.CAP_PROP_AUTO_WB] == 1 and cv2.CAP_PROP_WB_TEMPERATURE not in auto.calls

    path = tmp_path / "camera.json"
    assert CameraControls.load(path) is None
    c.save(path)
    assert CameraControls.load(path) == c


def test_software_white_balance_removes_blue_tint():
    from coffee_dataset.controls import CameraControls, ColorCorrectedCamera

    tinted = np.zeros((20, 20, 3), dtype=np.uint8)
    tinted[:, :] = (230, 190, 170)                      # BGR: bluish "white"
    b, g, r = CameraControls.gains_from_white_patch(tinted)
    assert b < 1.0 < r
    scale = 1.0 / max(b, 1.0, r)
    c = CameraControls(gain_b=b * scale, gain_g=g * scale, gain_r=r * scale)
    out = c.correct(tinted)
    spread = int(out[0, 0].max()) - int(out[0, 0].min())
    assert spread <= 3                                   # now (almost) neutral grey
    assert CameraControls().correct(tinted) is tinted    # no correction = untouched frame

    class Fake:
        def open(self): pass
        def read(self): return tinted
        def close(self): pass

    assert np.array_equal(ColorCorrectedCamera(Fake(), c).read(), out)


def test_old_camera_json_still_loads(tmp_path):
    from coffee_dataset.controls import CameraControls

    path = tmp_path / "camera.json"
    path.write_text('{"auto_wb": false, "wb_temperature": 6000}', encoding="utf-8")
    c = CameraControls.load(path)
    assert c.auto_wb is False and c.gain_b == 1.0


def test_sharpen_increases_edge_contrast_and_keeps_shape():
    from coffee_dataset.controls import CameraControls

    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:, 30:] = 160
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    out = CameraControls(sharpen=1.0).correct(blurred)
    assert out.shape == blurred.shape
    edge_before = int(blurred[30, 34, 0]) - int(blurred[30, 26, 0])
    edge_after = int(out[30, 34, 0]) - int(out[30, 26, 0])
    assert edge_after > edge_before
    assert CameraControls(sharpen=0.0).correct(blurred) is blurred


def test_camera_retries_plain_when_driver_returns_tiny_frames(monkeypatch):
    from coffee_dataset import cameras

    opened = []

    class FakeCap:
        def __init__(self, index, backend=None):
            self.compressed = False
            opened.append(self)

        def isOpened(self): return True
        def set(self, prop, value):
            if prop == cv2.CAP_PROP_FOURCC:
                self.compressed = True
            return True
        def read(self):
            size = (120, 160) if self.compressed else (720, 1280)    # tiny frames only in compressed mode
            return True, np.zeros((*size, 3), dtype=np.uint8)
        def release(self): pass
        def get(self, prop): return 0

    monkeypatch.setattr(cameras.cv2, "VideoCapture", FakeCap)
    cam = cameras.OpenCvCamera(0, 1280, 720, warmup_frames=2)
    cam.open()
    assert len(opened) == 2                       # retried without MJPG
    assert cam.read().shape[:2] == (720, 1280)


def test_crop_never_returns_a_tiny_image():
    from coffee_dataset.crop import CropRegion

    region = CropRegion(0.39, 0.0, 0.39, 0.69)
    tiny = np.zeros((120, 160, 3), dtype=np.uint8)       # camera that returned a tiny frame
    assert region.apply(tiny) is tiny
    normal = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert region.apply(normal).shape[0] >= 200
