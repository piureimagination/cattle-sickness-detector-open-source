"""FastAPI inference server for the cattle sickness (Lumpy Skin Disease) detector.

Serves a single binary classifier: healthy vs. Lumpy Skin Disease, from an
image of a cow. This is NOT a general-purpose cattle illness detector - see
README.md for scope and limitations.
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

load_dotenv()

MODEL_PATH = Path(__file__).parent.parent / "models" / "cattle_sickness_detection_best.keras"
IMAGE_SIZE = (224, 224)

# Predictions with a raw score in [LOW_CONFIDENCE, HIGH_CONFIDENCE] are too close
# to the decision boundary to report as a diagnosis; they're flagged as uncertain
# instead of forcing a binary call.
LOW_CONFIDENCE = 0.35
HIGH_CONFIDENCE = 0.65

model: tf.keras.Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    yield


app = FastAPI(title="Cattle Sickness Detection API", version="0.3.0", lifespan=lifespan)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into the [1, 224, 224, 3] float32 array the model expects.

    Pixels are kept in [0, 255]; EfficientNetV2B0 normalizes internally, so do
    not rescale here.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMAGE_SIZE)
    array = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(array, axis=0)


@app.get("/")
def home() -> dict:
    return {
        "message": "Cattle Sickness Detection API is online.",
        "docs_url": "/docs",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Run inference on an uploaded image and return a diagnosis."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await file.read()
    try:
        input_data = preprocess_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    prediction = float(model.predict(input_data, verbose=0)[0][0])

    if LOW_CONFIDENCE <= prediction <= HIGH_CONFIDENCE:
        diagnosis = "Uncertain"
        predicted_class_id = None
        confidence = 1.0 - abs(prediction - 0.5) * 2  # closeness to the 0.5 boundary
    elif prediction > HIGH_CONFIDENCE:
        diagnosis = "Lumpy Skin Disease"
        predicted_class_id = 1
        confidence = prediction
    else:
        diagnosis = "Healthy"
        predicted_class_id = 0
        confidence = 1.0 - prediction

    return {
        "filename": file.filename,
        "predicted_class_id": predicted_class_id,
        "diagnosis": diagnosis,
        "raw_score": round(prediction, 4),
        "confidence": round(confidence, 4),
        "uncertain": diagnosis == "Uncertain",
    }
