from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

from src.engine.trainer import run_training
from tests.helpers import make_cfg


def _tiny_compiled_model(img_size=(16, 16)):
    inputs = layers.Input(shape=(*img_size, 3))
    x = layers.GlobalAveragePooling2D()(inputs)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _tiny_dataset(img_size=(16, 16), n=8, batch_size=4):
    images = tf.random.uniform((n, *img_size, 3), maxval=255)
    labels = tf.cast(tf.random.uniform((n, 1), maxval=2, dtype=tf.int32), tf.float32)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)


def test_run_training_returns_merged_history_from_both_phases(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_SAVE_PATH", str(tmp_path / "models"))
    cfg = make_cfg(project_name="trainer_test", epochs_phase_1=1, epochs_phase_2=1, patience=1)
    model = _tiny_compiled_model()
    train_ds = _tiny_dataset()
    val_ds = _tiny_dataset()

    history = run_training(model, train_ds, val_ds, cfg)

    for key in ("loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc"):
        assert key in history.history
        # one epoch per phase -> two entries per metric
        assert len(history.history[key]) == 2


def test_run_training_saves_checkpoint_to_model_save_path(tmp_path, monkeypatch):
    save_dir = tmp_path / "models"
    monkeypatch.setenv("MODEL_SAVE_PATH", str(save_dir))
    cfg = make_cfg(project_name="checkpoint_test", epochs_phase_1=1, epochs_phase_2=1, patience=1)
    model = _tiny_compiled_model()

    run_training(model, _tiny_dataset(), _tiny_dataset(), cfg)

    expected = save_dir / "checkpoint_test_best.keras"
    assert expected.exists()


def test_run_training_reuses_same_checkpoint_across_phases(tmp_path, monkeypatch):
    """Regression test: a fresh ModelCheckpoint per phase forgets phase 1's
    best val_auc, so a worse phase-2 epoch can overwrite a better phase-1
    checkpoint just for beating phase 2's own from-scratch baseline. The
    checkpoint callback must be the same instance across both model.fit
    calls so its internal "best so far" persists across the phase boundary.
    """
    monkeypatch.setenv("MODEL_SAVE_PATH", str(tmp_path / "models"))
    cfg = make_cfg(project_name="shared_ckpt_test", epochs_phase_1=1, epochs_phase_2=1, patience=1)
    model = _tiny_compiled_model()

    seen_checkpoints = []
    real_fit = model.fit

    def spying_fit(*args, **kwargs):
        checkpoint = next(cb for cb in kwargs["callbacks"] if isinstance(cb, tf.keras.callbacks.ModelCheckpoint))
        seen_checkpoints.append(checkpoint)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(model, "fit", spying_fit)

    run_training(model, _tiny_dataset(), _tiny_dataset(), cfg)

    assert len(seen_checkpoints) == 2
    assert seen_checkpoints[0] is seen_checkpoints[1]


def test_run_training_defaults_save_path_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_SAVE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = make_cfg(project_name="default_path_test", epochs_phase_1=1, epochs_phase_2=1, patience=1)
    model = _tiny_compiled_model()

    run_training(model, _tiny_dataset(), _tiny_dataset(), cfg)

    assert (Path("models") / "default_path_test_best.keras").exists()
