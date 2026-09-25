import dataclasses
from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config

VALID_YAML = {
    "experiment": {
        "project_name": "cattle_sickness_detection",
        "seed": 42,
        "dataset": {
            "kaggle_dataset": "someuser/some-dataset",
            "base_dir": "data/processed",
            "raw_data_dir": "data/raw",
            "source_dirs": {"healthy": "Healthy Cows", "lumpy": "Lumpy Cows"},
        },
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        "model": {
            "img_size": [224, 224],
            "batch_size": 32,
            "learning_rate": 0.001,
            "learning_rate_phase_2": 0.00001,
            "dropout_rate": 0.5,
            "epochs_phase_1": 15,
            "epochs_phase_2": 10,
            "patience": 5,
        },
    }
}


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_config_reads_all_fields(tmp_path):
    cfg = load_config(str(_write_yaml(tmp_path, VALID_YAML)))

    assert cfg.seed == 42
    assert cfg.project_name == "cattle_sickness_detection"
    assert cfg.kaggle_dataset == "someuser/some-dataset"
    assert cfg.base_dir == Path("data/processed")
    assert cfg.raw_dir == Path("data/raw")
    assert cfg.source_dirs == {"healthy": "Healthy Cows", "lumpy": "Lumpy Cows"}
    assert cfg.train_ratio == 0.7
    assert cfg.val_ratio == 0.15
    assert cfg.test_ratio == 0.15
    assert cfg.img_size == (224, 224)
    assert isinstance(cfg.img_size, tuple)
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 0.001
    assert cfg.learning_rate_phase_2 == 0.00001
    assert cfg.dropout_rate == 0.5
    assert cfg.epochs_phase_1 == 15
    assert cfg.epochs_phase_2 == 10
    assert cfg.patience == 5


def test_load_config_defaults_raw_data_dir_when_missing(tmp_path):
    data = yaml.safe_load(yaml.dump(VALID_YAML))
    del data["experiment"]["dataset"]["raw_data_dir"]

    cfg = load_config(str(_write_yaml(tmp_path, data)))

    assert cfg.raw_dir == Path("data/raw")


def test_load_config_honors_custom_raw_data_dir(tmp_path):
    data = yaml.safe_load(yaml.dump(VALID_YAML))
    data["experiment"]["dataset"]["raw_data_dir"] = "custom/raw/path"

    cfg = load_config(str(_write_yaml(tmp_path, data)))

    assert cfg.raw_dir == Path("custom/raw/path")


def test_config_is_immutable(tmp_path):
    cfg = load_config(str(_write_yaml(tmp_path, VALID_YAML)))

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 0


def test_load_config_missing_required_key_raises(tmp_path):
    data = yaml.safe_load(yaml.dump(VALID_YAML))
    del data["experiment"]["model"]["learning_rate"]

    with pytest.raises(KeyError):
        load_config(str(_write_yaml(tmp_path, data)))
