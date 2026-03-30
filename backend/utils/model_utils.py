"""
utils/model_utils.py
--------------------
Helper functions for loading the trained CNN model and running predictions.
"""

import os
import numpy as np

# Class names must match the sub-folder names used during training
CLASS_NAMES = ["Acne", "Blackheads", "Dark Spots", "Normal", "Pores", "Wrinkles"]

# Absolute path to the saved model weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "skin_model.h5")

# Module-level cache so the model is loaded only once per server lifetime
_model = None


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

    _model = tf.keras.models.load_model(MODEL_PATH)
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

    # Index of the highest-probability class
    class_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_index])
    predicted_class = CLASS_NAMES[class_index]

    return predicted_class, confidence
