from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration loaded from config/pipeline.yaml."""

    seed: int
    project_name: str
    kaggle_dataset: str
    base_dir: Path
    raw_dir: Path
    source_dirs: dict
    train_ratio: float
    val_ratio: float
    test_ratio: float
    img_size: tuple
    batch_size: int
    learning_rate: float
    learning_rate_phase_2: float
    dropout_rate: float
    epochs_phase_1: int
    epochs_phase_2: int
    patience: int


def load_config(path: str = "config/pipeline.yaml") -> PipelineConfig:
    """Load and validate the pipeline config, returning an immutable PipelineConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    experiment = raw["experiment"]
    dataset = experiment["dataset"]
    model = experiment["model"]
    splits = experiment["splits"]

    return PipelineConfig(
        seed=experiment["seed"],
        project_name=experiment["project_name"],
        kaggle_dataset=dataset["kaggle_dataset"],
        base_dir=Path(dataset["base_dir"]),
        raw_dir=Path(dataset.get("raw_data_dir", "data/raw")),
        source_dirs=dataset["source_dirs"],
        train_ratio=splits["train"],
        val_ratio=splits["val"],
        test_ratio=splits["test"],
        img_size=tuple(model["img_size"]),
        batch_size=model["batch_size"],
        learning_rate=model["learning_rate"],
        learning_rate_phase_2=model["learning_rate_phase_2"],
        dropout_rate=model["dropout_rate"],
        epochs_phase_1=model["epochs_phase_1"],
        epochs_phase_2=model["epochs_phase_2"],
        patience=model["patience"],
    )
