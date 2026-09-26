"""Evaluate a trained model on the held-out test set with a full breakdown.

Reports a confusion matrix and per-class precision/recall/F1, not just
overall accuracy - useful for seeing whether errors are one-sided (e.g.
missing Lumpy cases more than Healthy ones) rather than evenly spread.

Usage:
    python scripts/evaluate.py                          # uses the best checkpoint from config
    python scripts/evaluate.py --model models/some.keras --threshold 0.5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import build_datasets
from src.utils.config import load_config

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to the pipeline config YAML.")
    parser.add_argument("--model", default=None, help="Path to a .keras model. Defaults to the best checkpoint named in config.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for Lumpy vs Healthy.")
    return parser.parse_args()


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_path = Path(args.model) if args.model else Path(os.environ.get("MODEL_SAVE_PATH", "models")) / f"{cfg.project_name}_best.keras"
    model = tf.keras.models.load_model(model_path)

    _, _, test_ds = build_datasets(cfg)

    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().reshape(-1).tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= args.threshold).astype(int)

    # Label convention: 0 = Healthy, 1 = Lumpy (alphabetical folder order, see src/data/pipeline.py)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision_lumpy = safe_div(tp, tp + fp)
    recall_lumpy = safe_div(tp, tp + fn)
    f1_lumpy = safe_div(2 * precision_lumpy * recall_lumpy, precision_lumpy + recall_lumpy)

    precision_healthy = safe_div(tn, tn + fn)
    recall_healthy = safe_div(tn, tn + fp)
    f1_healthy = safe_div(2 * precision_healthy * recall_healthy, precision_healthy + recall_healthy)

    accuracy = safe_div(tp + tn, len(y_true))

    print(f"\nModel: {model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Test set: {len(y_true)} images ({int(np.sum(y_true == 0))} healthy, {int(np.sum(y_true == 1))} lumpy)\n")

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"{'':>18}{'Healthy':>10}{'Lumpy':>10}")
    print(f"{'Actual Healthy':>18}{tn:>10}{fp:>10}")
    print(f"{'Actual Lumpy':>18}{fn:>10}{tp:>10}\n")

    print(f"{'Class':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print(f"{'Healthy':>10}{precision_healthy:>12.3f}{recall_healthy:>10.3f}{f1_healthy:>10.3f}")
    print(f"{'Lumpy':>10}{precision_lumpy:>12.3f}{recall_lumpy:>10.3f}{f1_lumpy:>10.3f}\n")

    print(f"Overall accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
