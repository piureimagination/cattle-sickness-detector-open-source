from pathlib import Path

import pytest

from src.data.splitter import create_splits
from tests.helpers import make_cfg, make_tiny_image


def _populate_raw(raw_dir: Path, healthy: int, lumpy: int) -> None:
    for i in range(healthy):
        make_tiny_image(raw_dir / "healthy" / f"h{i}.jpg", seed=i)
    for i in range(lumpy):
        make_tiny_image(raw_dir / "lumpy" / f"l{i}.jpg", seed=100 + i)


def test_create_splits_matches_configured_ratios(tmp_path):
    raw_dir = tmp_path / "raw"
    _populate_raw(raw_dir, healthy=20, lumpy=20)
    cfg = make_cfg(base_dir=tmp_path / "processed", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    create_splits(raw_dir, cfg)

    for cls in ("healthy", "lumpy"):
        train = list((cfg.base_dir / "train" / cls).iterdir())
        val = list((cfg.base_dir / "val" / cls).iterdir())
        test = list((cfg.base_dir / "test" / cls).iterdir())
        assert len(train) == 14
        assert len(val) == 3
        assert len(test) == 3
        # no overlap between splits
        names = {f.name for f in train} | {f.name for f in val} | {f.name for f in test}
        assert len(names) == 20


def test_create_splits_is_deterministic_for_a_given_seed(tmp_path):
    raw_dir = tmp_path / "raw"
    _populate_raw(raw_dir, healthy=10, lumpy=10)
    cfg = make_cfg(base_dir=tmp_path / "processed_a", seed=99)
    create_splits(raw_dir, cfg)
    first_train = sorted(f.name for f in (cfg.base_dir / "train" / "healthy").iterdir())

    cfg2 = make_cfg(base_dir=tmp_path / "processed_b", seed=99)
    create_splits(raw_dir, cfg2)
    second_train = sorted(f.name for f in (cfg2.base_dir / "train" / "healthy").iterdir())

    assert first_train == second_train


def test_create_splits_rejects_bad_ratios(tmp_path):
    raw_dir = tmp_path / "raw"
    _populate_raw(raw_dir, healthy=5, lumpy=5)
    cfg = make_cfg(base_dir=tmp_path / "processed", train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    with pytest.raises(ValueError, match="sum to 1.0"):
        create_splits(raw_dir, cfg)


def test_create_splits_ignores_non_image_files(tmp_path):
    raw_dir = tmp_path / "raw"
    _populate_raw(raw_dir, healthy=4, lumpy=4)
    (raw_dir / "healthy" / "notes.txt").write_text("not an image")
    cfg = make_cfg(base_dir=tmp_path / "processed")

    create_splits(raw_dir, cfg)

    total = sum(
        len(list((cfg.base_dir / split / "healthy").iterdir())) for split in ("train", "val", "test")
    )
    assert total == 4


def test_create_splits_raises_when_nothing_to_process(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = make_cfg(base_dir=tmp_path / "processed")

    with pytest.raises(ValueError, match="No images were processed"):
        create_splits(raw_dir, cfg)


def test_create_splits_rebuilds_processed_dir_from_scratch(tmp_path):
    raw_dir = tmp_path / "raw"
    _populate_raw(raw_dir, healthy=4, lumpy=4)
    cfg = make_cfg(base_dir=tmp_path / "processed")

    create_splits(raw_dir, cfg)
    stale_marker = cfg.base_dir / "train" / "healthy" / "stale_leftover.jpg"
    make_tiny_image(stale_marker)
    assert stale_marker.exists()

    create_splits(raw_dir, cfg)

    assert not stale_marker.exists()
