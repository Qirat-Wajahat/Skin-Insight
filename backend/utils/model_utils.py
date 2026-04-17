"""
utils/model_utils.py
--------------------
Helper functions for loading the trained CNN model and running predictions.
"""

import os
import json
import numpy as np

# Class names must match the sub-folder names used during training.
# Prefer loading them from disk (saved by scripts/train_model.py).
DEFAULT_CLASS_NAMES = [
    "Acne",
    "Blackheads",
    "Combination",
    "Dark Circles",
    "Dark Spots",
    "Dry",
    "Normal",
    "Oily",
    "Pores",
    "Wrinkles",
]

# Absolute path to the saved model weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "skin_model.h5")
# Prefer the canonical list saved alongside the dataset/training pipeline.
# (A copy under backend/models is also supported if you add one.)
CLASS_NAMES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "class_names.json"
)

# Module-level cache so the model is loaded only once per server lifetime
_model = None
_class_names = None


def get_class_names() -> list[str]:
    """Return the class-name list used by the trained model."""
    global _class_names

    if _class_names is not None:
        return _class_names

    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                _class_names = data
                return _class_names
        except Exception:
            pass

    _class_names = DEFAULT_CLASS_NAMES
    return _class_names


def load_model():
    """
    Load the Keras model from disk, caching it for subsequent calls.

    Returns
    -------
    keras.Model
        The loaded (and compiled) CNN model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at MODEL_PATH.
    """
    global _model

    if _model is not None:
        return _model  # Return cached model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please train the model first by running: python scripts/train_model.py"
        )

    # Import TensorFlow only when needed to keep startup fast if model isn't used
    import tensorflow as tf

    # For inference we do not need the training-time compilation state.
    # Loading with `compile=False` avoids legacy H5 training-config parsing
    # issues (common when mixing older saved models with newer Keras).
    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def predict(image_array: np.ndarray) -> tuple[str, float]:
    """
    Run inference on a preprocessed image array.

    Parameters
    ----------
    image_array : np.ndarray
        Shape (1, 224, 224, 3), values normalised to [0, 1].

    Returns
    -------
    tuple[str, float]
        (predicted_class_name, confidence_score)
        confidence_score is in the range [0, 1].
    """
    model = load_model()

    # model.predict returns an array of shape (1, num_classes)
    predictions = model.predict(image_array, verbose=0)

    class_names = get_class_names()
    num_classes = int(predictions.shape[-1])
    if num_classes != len(class_names):
        raise RuntimeError(
            "Model output size does not match class_names. "
            f"Model outputs {num_classes} classes but class_names has {len(class_names)}. "
            "Re-train the model and ensure backend/models/class_names.json matches."
        )

    # Index of the highest-probability class
    class_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_index])
    predicted_class = class_names[class_index]

    return predicted_class, confidence
