import logging
import os
import random

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def lock_seeds(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    logger.info("Random seeds locked to %d.", seed)
