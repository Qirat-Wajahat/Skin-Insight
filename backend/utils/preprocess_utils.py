"""
utils/preprocess_utils.py
--------------------------
Helper functions for image preprocessing before sending to the CNN model.
"""

import numpy as np
from PIL import Image
import io

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # Must match the CNN input size


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a normalised numpy array ready for inference.

    Steps
    -----
    1. Open the image from bytes using Pillow.
    2. Convert to RGB (handles PNG with alpha channel, greyscale, etc.).
    3. Resize to IMAGE_SIZE (224×224).
    4. Convert to float32 numpy array.
    5. Normalise pixel values from [0, 255] → [0, 1].
    6. Add a batch dimension → shape (1, 224, 224, 3).

    Parameters
    ----------
    file_bytes : bytes
        Raw binary content of the uploaded image file.

    Returns
    -------
    np.ndarray
        Shape (1, 224, 224, 3), dtype float32, values in [0, 1].
    """
    # Open image and ensure it is in RGB colour space
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Resize to the size expected by the CNN
    image = image.resize(IMAGE_SIZE)

    # Convert to numpy array and cast to float32
    arr = np.array(image, dtype=np.float32)

    # Normalise pixel values to [0, 1]
    arr /= 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr
