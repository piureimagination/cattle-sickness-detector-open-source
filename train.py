"""Training entry point for the cattle sickness (Lumpy Skin Disease) detector.

Wires together the full pipeline: seed locking, dataset download, train/val/test
splitting, tf.data pipeline construction, model building, and two-phase training
(frozen backbone, then fine-tuning). Metrics are logged to MLflow.

Usage:
    python train.py
    python train.py --skip-download            # reuse an existing data/raw/
    python train.py --skip-download --skip-split  # reuse an existing data/processed/
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from src.data.download import download_data
from src.data.pipeline import build_datasets
from src.data.splitter import create_splits
from src.engine.trainer import run_training
from src.models.builder import build_model
from src.utils.config import load_config
from src.utils.seeds import lock_seeds

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to the pipeline config YAML.")
    parser.add_argument("--skip-download", action="store_true", help="Reuse the existing raw dataset instead of downloading from Kaggle.")
    parser.add_argument("--skip-split", action="store_true", help="Reuse the existing train/val/test split instead of rebuilding it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    lock_seeds(cfg.seed)

    if args.skip_download:
        logger.info("Skipping download, reusing raw data at %s", cfg.raw_dir)
    else:
        download_data(cfg.kaggle_dataset, cfg.raw_dir)

    if args.skip_split:
        logger.info("Skipping split, reusing processed data at %s", cfg.base_dir)
    else:
        create_splits(cfg.raw_dir, cfg)

    train_ds, val_ds, test_ds = build_datasets(cfg)
    model = build_model(cfg)

    mlflow.set_experiment(cfg.project_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "img_size": cfg.img_size,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "learning_rate_phase_2": cfg.learning_rate_phase_2,
                "dropout_rate": cfg.dropout_rate,
                "epochs_phase_1": cfg.epochs_phase_1,
                "epochs_phase_2": cfg.epochs_phase_2,
                "patience": cfg.patience,
            }
        )

        history = run_training(model, train_ds, val_ds, cfg)
        mlflow.log_metrics({key: values[-1] for key, values in history.history.items()})

        test_loss, test_accuracy, test_auc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy, "test_auc": test_auc})
        logger.info("Test set - loss: %.4f accuracy: %.4f auc: %.4f", test_loss, test_accuracy, test_auc)

        save_dir = Path(os.environ.get("MODEL_SAVE_PATH", "models"))
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / f"{cfg.project_name}_final.keras"
        model.save(final_path)
        logger.info("Final model saved to %s", final_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
