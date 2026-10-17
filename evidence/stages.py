"""Where the photos live: one entry per stage of the pipeline (data only, no logic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # coffee-roast-classifier
RECORDER_ROOT = PROJECT_ROOT.parent / "coffee-dataset"          # where the photos were recorded


@dataclass(frozen=True)
class Stage:
    key: str
    phase: str
    title: str
    folder: Path
    kind: str            # "flat": <folder>/<class>/*.jpg    "split": <folder>/<train|val|test>/<class>/*.jpg
    note: str = ""


STAGES: tuple[Stage, ...] = (
    # ---- Phase 1: recording ------------------------------------------------------------
    Stage("A0", "1. Recording", "Initial recording (not used by the models)",
          RECORDER_ROOT / "dataset" / "raw", "flat", "only claro and verde, first trial"),
    Stage("A1", "1. Recording", "Recording, first background (closed box)",
          RECORDER_ROOT / "dataset_closed" / "raw", "flat", "all classes, every photo taken"),
    Stage("A2", "1. Recording", "Recording, white sheet (kept after removing bursts)",
          RECORDER_ROOT / "dataset_white" / "raw", "flat", "photos closer than 0.95 s were moved aside"),
    Stage("A2s", "1. Recording", "White sheet: moved aside, nothing deleted",
          RECORDER_ROOT / "dataset_white" / "skipped", "flat", "bursts, duplicates and black frames"),
    # ---- Phase 2: filtering and cropping ----------------------------------------------
    Stage("B1", "2. Filtering and cropping", "First background after automatic filter",
          RECORDER_ROOT / "dataset_closed" / "curated", "flat", "hands, gloves, blur, mixed roast removed"),
    Stage("B2", "2. Filtering and cropping", "First background after crop (model 1 dataset)",
          RECORDER_ROOT / "dataset_final" / "curated", "flat", "centre 450x450 crop"),
    Stage("B3", "2. Filtering and cropping", "White sheet after automatic filter",
          RECORDER_ROOT / "dataset_white" / "curated", "flat"),
    Stage("B4", "2. Filtering and cropping", "White sheet after crop",
          RECORDER_ROOT / "dataset_white_final" / "curated", "flat"),
    # ---- Phase 3: merge and split ------------------------------------------------------
    Stage("C1", "3. Merge and split", "Merged dataset (both backgrounds)",
          RECORDER_ROOT / "dataset_v2" / "curated", "flat", "B2 + B4"),
    Stage("C2", "3. Merge and split", "Model 1 split",
          RECORDER_ROOT / "dataset_final" / "split", "split", "blocks of 5 consecutive photos"),
    Stage("C3", "3. Merge and split", "Model 2 split",
          RECORDER_ROOT / "dataset_v2" / "split_final", "split", "blocks of 5 consecutive photos"),
    # ---- Phase 4: Teachable Machine and the classifier project -------------------------
    Stage("D1", "4. Training and evaluation", "Uploaded to Teachable Machine (model 2)",
          RECORDER_ROOT / "tm_upload", "flat", "train photos only, 1 exact duplicate omitted"),
    Stage("D2", "4. Training and evaluation", "Classifier project data (model 2)",
          PROJECT_ROOT / "data", "split", "val and test were never uploaded"),
)
