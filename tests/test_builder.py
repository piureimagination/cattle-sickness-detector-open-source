import tensorflow as tf
from tensorflow.keras import layers, models

from src.models.builder import build_model
from tests.helpers import make_cfg


def _fake_efficientnet(input_shape, include_top, weights):
    """Stand-in for EfficientNetV2B0 that needs no network access.

    build_model() only relies on the backbone being a callable tf.keras.Model
    with a `trainable` flag and a `layers` list, so a tiny conv stack is
    enough to exercise the rest of the architecture wiring.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(4, 3, padding="same")(inputs)
    return models.Model(inputs, x, name="fake_efficientnet")


def test_build_model_output_shape_and_compile(monkeypatch):
    monkeypatch.setattr(tf.keras.applications, "EfficientNetV2B0", _fake_efficientnet)
    cfg = make_cfg(img_size=(32, 32), batch_size=2)

    model = build_model(cfg)

    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 1)
    assert model.loss is not None

    # model.metrics_names collapses to a generic "compile_metrics" placeholder
    # in this Keras version; return_dict=True is what actually surfaces the
    # names each metric was constructed with.
    results = model.test_on_batch(tf.zeros((2, 32, 32, 3)), tf.zeros((2, 1)), return_dict=True)
    assert "accuracy" in results
    assert "auc" in results


def test_build_model_backbone_starts_frozen(monkeypatch):
    monkeypatch.setattr(tf.keras.applications, "EfficientNetV2B0", _fake_efficientnet)
    cfg = make_cfg(img_size=(32, 32))

    model = build_model(cfg)

    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    assert backbone.trainable is False


def test_build_model_runs_a_forward_pass(monkeypatch):
    monkeypatch.setattr(tf.keras.applications, "EfficientNetV2B0", _fake_efficientnet)
    cfg = make_cfg(img_size=(32, 32), batch_size=2)

    model = build_model(cfg)
    dummy_input = tf.zeros((2, 32, 32, 3))
    output = model(dummy_input, training=False)

    assert output.shape == (2, 1)
    assert bool(tf.reduce_all((output >= 0) & (output <= 1)))
