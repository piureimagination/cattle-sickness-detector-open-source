import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(cfg) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build optimized tf.data pipelines for the train, val and test splits.

    Pixels are kept as float32 in [0, 255]; EfficientNetV2B0 normalizes
    internally, so no rescaling happens here.
    """
    logger.info("Building tf.data pipelines...")

    def create_ds(subset: str, shuffle: bool = False) -> tf.data.Dataset:
        ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(cfg.base_dir / subset),
            labels="inferred",
            label_mode="binary",
            batch_size=cfg.batch_size,
            image_size=cfg.img_size,
            shuffle=shuffle,
            seed=cfg.seed if shuffle else None,
        )
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        if shuffle:
            # Shuffle after cache so shuffling operates on fast in-memory data.
            ds = ds.shuffle(buffer_size=1000, seed=cfg.seed)
        return ds.prefetch(buffer_size=AUTOTUNE)

    train_ds = create_ds("train", shuffle=True)
    val_ds = create_ds("val", shuffle=False)
    test_ds = create_ds("test", shuffle=False)

    logger.info("Datasets built.")
    return train_ds, val_ds, test_ds
