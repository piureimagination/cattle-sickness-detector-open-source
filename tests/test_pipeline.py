import numpy as np
import tensorflow as tf

from src.data.pipeline import build_datasets
from tests.helpers import make_cfg, make_tiny_image


def _populate_split(base_dir, split, cls, count, seed_offset):
    for i in range(count):
        make_tiny_image(base_dir / split / cls / f"{cls}_{i}.jpg", size=(32, 32), seed=seed_offset + i)


def _build_ready_cfg(tmp_path, batch_size=2):
    base_dir = tmp_path / "processed"
    for split in ("train", "val", "test"):
        _populate_split(base_dir, split, "healthy", 4, seed_offset=0)
        _populate_split(base_dir, split, "lumpy", 4, seed_offset=50)
    return make_cfg(base_dir=base_dir, batch_size=batch_size, img_size=(32, 32))


def test_build_datasets_returns_three_splits_with_expected_shapes(tmp_path):
    cfg = _build_ready_cfg(tmp_path)

    train_ds, val_ds, test_ds = build_datasets(cfg)

    for ds in (train_ds, val_ds, test_ds):
        images, labels = next(iter(ds))
        assert images.shape[1:] == (32, 32, 3)
        assert labels.shape[1:] == (1,)
        assert images.dtype == tf.float32


def test_build_datasets_keeps_pixels_unscaled(tmp_path):
    # Pixels must stay in [0, 255] - EfficientNetV2B0 normalizes internally,
    # so this pipeline must NOT rescale to [0, 1].
    cfg = _build_ready_cfg(tmp_path)

    train_ds, _, _ = build_datasets(cfg)
    images, _ = next(iter(train_ds))

    assert float(tf.reduce_max(images)) > 1.0


def test_build_datasets_respects_batch_size(tmp_path):
    cfg = _build_ready_cfg(tmp_path, batch_size=3)

    train_ds, _, _ = build_datasets(cfg)
    images, labels = next(iter(train_ds))

    assert images.shape[0] == 3
    assert labels.shape[0] == 3


def test_build_datasets_train_is_shuffled_deterministically_by_seed(tmp_path):
    cfg_a = _build_ready_cfg(tmp_path)
    cfg_b = make_cfg(base_dir=cfg_a.base_dir, batch_size=cfg_a.batch_size, img_size=cfg_a.img_size, seed=cfg_a.seed)

    train_a, _, _ = build_datasets(cfg_a)
    train_b, _, _ = build_datasets(cfg_b)

    images_a = np.concatenate([b[0].numpy() for b in train_a], axis=0)
    images_b = np.concatenate([b[0].numpy() for b in train_b], axis=0)

    assert np.array_equal(images_a, images_b)
