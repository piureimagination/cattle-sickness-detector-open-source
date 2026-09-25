import logging

import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


def build_model(cfg) -> tf.keras.Model:
    """Build an EfficientNetV2B0 transfer-learning binary classifier.

    Architecture: Input -> Augmentation -> EfficientNetV2B0 (frozen) ->
    GlobalAveragePooling -> BatchNorm -> Dropout -> Dense(128) -> Dense(1, sigmoid)

    EfficientNetV2B0 expects raw [0, 255] float32 input and normalizes
    internally. Augmentation layers are active only during training.
    """
    logger.info("Building EfficientNetV2B0 architecture...")

    inputs = layers.Input(shape=(*cfg.img_size, 3))

    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)

    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(*cfg.img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model(x, training=False)  # training=False keeps BatchNorm frozen

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(cfg.dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="Cattle_Sickness_Detection_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info("Model built. Total params: %s", f"{model.count_params():,}")
    return model
