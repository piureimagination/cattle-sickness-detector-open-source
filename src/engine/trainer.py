import logging
import os
from pathlib import Path

import tensorflow as tf

logger = logging.getLogger(__name__)


def run_training(model, train_ds, val_ds, cfg):
    """Train in two phases: a frozen-backbone phase, then fine-tuning.

    Phase 1 trains only the classification head. Phase 2 unfreezes the last
    30 layers of the backbone and continues training at a lower learning rate.
    Returns the merged Keras History from both phases.
    """
    save_dir = Path(os.environ.get("MODEL_SAVE_PATH", "models"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f"{cfg.project_name}_best.keras"

    def callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, cfg.patience // 2),
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            ),
        ]

    logger.info("Phase 1: training classification head...")
    phase1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs_phase_1,
        callbacks=callbacks(),
    )

    logger.info("Phase 2: fine-tuning backbone...")
    base_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    if base_model is not None:
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate_phase_2),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    phase2_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs_phase_2,
        callbacks=callbacks(),
    )

    logger.info("Training complete.")

    # Merge phase 2 history into phase 1 so callers see one continuous record.
    for key, values in phase2_history.history.items():
        phase1_history.history[key].extend(values)

    return phase1_history
