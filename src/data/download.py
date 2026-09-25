import logging
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)

HEALTHY_KEYWORDS = ("healthy", "normal")
LUMPY_KEYWORDS = ("lumpy", "infected", "diseased")


def normalize_dataset(dataset_path: Path, raw_dir: Path) -> None:
    """Copy a downloaded Kaggle dataset into raw_dir/{healthy,lumpy}/.

    Labels are inferred from keywords in each file's path, since the source
    dataset ships folder names rather than a label file. Raises if any file
    can't be classified, rather than guessing - a silent mislabel here would
    quietly corrupt training data.
    """
    logger.info("Normalizing dataset into %s", raw_dir)

    healthy_dir = raw_dir / "healthy"
    lumpy_dir = raw_dir / "lumpy"
    healthy_dir.mkdir(parents=True, exist_ok=True)
    lumpy_dir.mkdir(parents=True, exist_ok=True)

    all_files = [f for f in dataset_path.rglob("*") if f.is_file()]
    if not all_files:
        raise ValueError(f"No files found in downloaded dataset at {dataset_path}")

    seen_names = set()
    unmatched = []
    count_healthy, count_lumpy = 0, 0

    for file in all_files:
        if file.name in seen_names:
            continue
        seen_names.add(file.name)

        path_lower = str(file).lower()
        if any(keyword in path_lower for keyword in HEALTHY_KEYWORDS):
            shutil.copy(file, healthy_dir / file.name)
            count_healthy += 1
        elif any(keyword in path_lower for keyword in LUMPY_KEYWORDS):
            shutil.copy(file, lumpy_dir / file.name)
            count_lumpy += 1
        else:
            unmatched.append(str(file))

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} file(s) matched neither healthy nor lumpy keywords "
            f"(e.g. {unmatched[0]}). Refusing to guess labels - update "
            "HEALTHY_KEYWORDS/LUMPY_KEYWORDS in src/data/download.py to match "
            "the dataset's actual naming, or remove the unrecognized files."
        )
    if count_healthy == 0 or count_lumpy == 0:
        raise ValueError(
            f"Normalization produced an empty class (healthy={count_healthy}, "
            f"lumpy={count_lumpy}); check the dataset structure."
        )

    logger.info("Normalized dataset: healthy=%d lumpy=%d", count_healthy, count_lumpy)


def download_data(dataset_id: str, raw_dir: Path) -> Path:
    """Download a Kaggle dataset and normalize it into raw_dir/{healthy,lumpy}/."""
    logger.info("Downloading dataset: %s", dataset_id)

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        logger.info("Cleared existing raw directory: %s", raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download(dataset_id))
    logger.info("Dataset downloaded to %s", dataset_path)

    normalize_dataset(dataset_path, raw_dir)
    return raw_dir