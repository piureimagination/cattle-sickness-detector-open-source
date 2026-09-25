# Cattle Lumpy Skin Disease Detection

A binary image classifier that flags **Lumpy Skin Disease (LSD) vs. Healthy** in
cattle from a photo, served behind a small FastAPI inference API. Built with a
transfer-learning EfficientNetV2B0 backbone on top of TensorFlow/Keras.

## Scope and limitations

This model detects **one specific disease** (Lumpy Skin Disease) via a binary
classifier - it is not a general cattle health or "sickness" detector, and it
was not trained to recognize any other condition. A photo of a healthy animal,
an animal with a different illness, or a non-cattle subject will still get a
Healthy/LSD answer; the API reports predictions inside `[0.35, 0.65]` as
`"Uncertain"` rather than forcing a call, but this is not a substitute for
veterinary diagnosis. Treat all output as a screening aid, not a diagnosis.

## Architecture

```
Input (224x224x3)
  -> data augmentation (flip / rotate / zoom, training only)
  -> EfficientNetV2B0 backbone (ImageNet weights, frozen in phase 1)
  -> GlobalAveragePooling -> BatchNorm -> Dropout -> Dense(128) -> Dense(1, sigmoid)
```

Training runs in two phases: phase 1 trains only the classification head with
the backbone frozen; phase 2 unfreezes the backbone's last 30 layers and
fine-tunes at a lower learning rate. See `src/models/builder.py` and
`src/engine/trainer.py`.

## Project layout

```
config/pipeline.yaml   Experiment config (dataset, splits, hyperparameters)
src/data/               Dataset download, labeling, and train/val/test splitting
src/models/builder.py   Model architecture
src/engine/trainer.py   Two-phase training loop
src/utils/              Config loading, seed locking
src/main.py             FastAPI inference server
train.py                Training entry point (CLI)
tests/                  Unit tests (see Testing below)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # + test dependencies (pytest, httpx)
cp .env.example .env                   # fill in your own Kaggle API token
```

## Training

```bash
python train.py                              # full run: download, split, train
python train.py --skip-download               # reuse an existing data/raw/
python train.py --skip-download --skip-split  # reuse an existing data/processed/
```

Hyperparameters and the Kaggle dataset ID live in `config/pipeline.yaml`.
Runs are logged to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`
to view them locally); the best checkpoint is saved to
`${MODEL_SAVE_PATH}/{project_name}_best.keras` and the end-of-run model to
`${MODEL_SAVE_PATH}/{project_name}_final.keras`.

## Serving

```bash
uvicorn src.main:app --reload
```

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/cow_photo.jpg"
```

```json
{
  "filename": "cow_photo.jpg",
  "predicted_class_id": 1,
  "diagnosis": "Lumpy Skin Disease",
  "raw_score": 0.8123,
  "confidence": 0.8123,
  "uncertain": false
}
```

## Dataset

Training data comes from the Kaggle dataset
[`shivamagarwal29/cow-lumpy-disease-dataset`](https://www.kaggle.com/datasets/shivamagarwal29/cow-lumpy-disease-dataset).
No license is listed on the dataset page at the time of writing - check its
current terms yourself before redistributing data or model weights trained on
it. For that reason this repo ships code only; model weights are not included
in version control (see `.gitignore`) and should be distributed separately
(a GitHub Release asset, Git LFS, or a model hub) with a decision you make
after reviewing the dataset's license.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is unit-level and runs in seconds with no GPU, network, or Kaggle
credentials required:

- `test_config.py`, `test_seeds.py` - config loading/validation, seed reproducibility
- `test_download.py` - dataset labeling and the fail-loudly-on-unrecognized-file behavior
- `test_splitter.py` - deterministic train/val/test splitting
- `test_pipeline.py` - tf.data pipeline shapes, dtype, and unscaled [0, 255] pixel range
- `test_builder.py` - model architecture and compile config, with the EfficientNetV2B0
  backbone mocked out so tests don't need network access to download ImageNet weights
- `test_trainer.py` - the two-phase training loop and checkpoint saving, on tiny synthetic data
- `test_main.py` - the `/predict` API, including the confidence-band boundaries, using a
  small model with fixed weights so predictions are deterministic

`tests/tests.ipynb` is a separate, manual notebook for checking GPU availability in a
Colab-style environment - it's not part of the automated suite.

## License

Code is released under the [MIT License](LICENSE). This does not cover the
Kaggle dataset or any model weights trained on it - see Dataset above.
