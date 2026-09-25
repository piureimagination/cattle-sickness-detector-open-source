import io

import pytest
import tensorflow as tf
from fastapi.testclient import TestClient
from PIL import Image
from tensorflow.keras import layers, models

import src.main as main_module
from tests.helpers import logit


def _make_constant_model(output_value: float, path):
    """A model that always predicts `output_value`, regardless of input.

    Kernel is zero-initialized so the input is ignored; the bias is set so
    sigmoid(bias) == output_value. This makes the confidence-band logic in
    /predict testable exactly, without depending on real trained weights.
    """
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.GlobalAveragePooling2D()(inputs)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(logit(output_value)),
    )(x)
    model = models.Model(inputs, outputs)
    model.save(path)
    return path


def _client_with_model(tmp_path, monkeypatch, output_value):
    model_path = _make_constant_model(output_value, tmp_path / "model.keras")
    monkeypatch.setattr(main_module, "MODEL_PATH", model_path)
    return TestClient(main_module.app)


def _sample_image_bytes():
    image = Image.fromarray((tf.zeros((64, 64, 3), dtype=tf.uint8)).numpy())
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_root_endpoint(tmp_path, monkeypatch):
    with _client_with_model(tmp_path, monkeypatch, 0.5) as client:
        response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"]
    assert body["docs_url"] == "/docs"


def test_predict_healthy_below_low_confidence(tmp_path, monkeypatch):
    with _client_with_model(tmp_path, monkeypatch, 0.1) as client:
        response = client.post("/predict", files={"file": ("cow.jpg", _sample_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["diagnosis"] == "Healthy"
    assert body["predicted_class_id"] == 0
    assert body["uncertain"] is False


def test_predict_lumpy_above_high_confidence(tmp_path, monkeypatch):
    with _client_with_model(tmp_path, monkeypatch, 0.9) as client:
        response = client.post("/predict", files={"file": ("cow.jpg", _sample_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["diagnosis"] == "Lumpy Skin Disease"
    assert body["predicted_class_id"] == 1
    assert body["uncertain"] is False


def test_predict_uncertain_band(tmp_path, monkeypatch):
    with _client_with_model(tmp_path, monkeypatch, 0.5) as client:
        response = client.post("/predict", files={"file": ("cow.jpg", _sample_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["diagnosis"] == "Uncertain"
    assert body["predicted_class_id"] is None
    assert body["uncertain"] is True


@pytest.mark.parametrize("boundary_value", [0.36, 0.64])
def test_predict_is_uncertain_just_inside_the_band(tmp_path, monkeypatch, boundary_value):
    # Deliberately not testing the exact 0.35/0.65 edge: the constant-output
    # model's prediction goes through a real sigmoid(bias) in float32, so an
    # exact boundary value can land a hair on either side depending on
    # rounding. A value clearly inside the band is what actually matters.
    with _client_with_model(tmp_path, monkeypatch, boundary_value) as client:
        response = client.post("/predict", files={"file": ("cow.jpg", _sample_image_bytes(), "image/jpeg")})

    assert response.json()["diagnosis"] == "Uncertain"


@pytest.mark.parametrize("just_outside_value", [0.34, 0.66])
def test_predict_is_decisive_just_outside_the_band(tmp_path, monkeypatch, just_outside_value):
    with _client_with_model(tmp_path, monkeypatch, just_outside_value) as client:
        response = client.post("/predict", files={"file": ("cow.jpg", _sample_image_bytes(), "image/jpeg")})

    assert response.json()["diagnosis"] != "Uncertain"


def test_predict_rejects_non_image_content_type(tmp_path, monkeypatch):
    with _client_with_model(tmp_path, monkeypatch, 0.5) as client:
        response = client.post(
            "/predict", files={"file": ("notes.txt", io.BytesIO(b"not an image"), "text/plain")}
        )

    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes(tmp_path, monkeypatch):
    with _client_with_model(tmp_path, monkeypatch, 0.5) as client:
        response = client.post(
            "/predict",
            files={"file": ("cow.jpg", io.BytesIO(b"this is not a real jpeg"), "image/jpeg")},
        )

    assert response.status_code == 400
