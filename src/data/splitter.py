import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASSES = ("healthy", "lumpy")


def create_splits(root_raw_data: Path, cfg) -> None:
    """Split raw_dir/{healthy,lumpy}/ into a deterministic train/val/test layout.

    Rebuilds cfg.base_dir from scratch each run so repeated calls don't
    accumulate stale files from a previous split.
    """
    logger.info("Creating dataset splits from %s", root_raw_data)

    raw_dir = Path(root_raw_data)
    processed_dir = Path(cfg.base_dir)

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        logger.info("Cleared existing processed directory: %s", processed_dir)

    train_ratio = getattr(cfg, "train_ratio", 0.7)
    val_ratio = getattr(cfg, "val_ratio", 0.15)
    test_ratio = getattr(cfg, "test_ratio", 0.15)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    random.seed(getattr(cfg, "seed", 42))

    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    total_processed = 0
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            logger.warning("Missing class folder, skipping: %s", cls_dir)
            continue

        files = [f for f in cls_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            logger.warning("No images found for class %s in %s", cls, cls_dir)
            continue

        random.shuffle(files)
        n_train = int(len(files) * train_ratio)
        n_val = int(len(files) * val_ratio)
        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            for f in split_files:
                shutil.copy2(f, processed_dir / split_name / cls / f.name)

        logger.info(
            "%s: total=%d train=%d val=%d test=%d",
            cls,
            len(files),
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )
        total_processed += len(files)

    if total_processed == 0:
        raise ValueError("No images were processed - check raw_dir contents.")

    logger.info("Split complete: %d images processed", total_processed)
