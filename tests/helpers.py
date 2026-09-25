"""Shared test helpers: config factory and tiny synthetic assets."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import PipelineConfig


def make_cfg(**overrides) -> PipelineConfig:
    """Build a small, fast PipelineConfig for tests, with overridable fields."""
    defaults = dict(
        seed=42,
        project_name="test_project",
        kaggle_dataset="unused/for-tests",
        base_dir=Path("data/processed"),
        raw_dir=Path("data/raw"),
        source_dirs={"healthy": "Healthy Cows", "lumpy": "Lumpy Cows"},
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        img_size=(32, 32),
        batch_size=2,
        learning_rate=0.001,
        learning_rate_phase_2=0.00001,
        dropout_rate=0.5,
        epochs_phase_1=1,
        epochs_phase_2=1,
        patience=1,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def make_tiny_image(path: Path, size: tuple[int, int] = (32, 32), seed: int = 0) -> None:
    """Write a small random RGB JPEG to path, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def logit(probability: float) -> float:
    """Inverse sigmoid, for building models with a known constant output."""
    return math.log(probability / (1 - probability))
