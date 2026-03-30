"""
preprocess.py
-------------
Loads images from datasets/train/ and datasets/test/, resizes them to 224x224,
normalizes pixel values to [0, 1], and prepares batched datasets for model training.

Expected directory structure:
    datasets/
        train/
            Acne/
            Wrinkles/
            Blackheads/
            Dark Spots/
            Pores/
            Normal/
        test/
            Acne/
            ...
"""

import os
import tensorflow as tf

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # Input size expected by the CNN
BATCH_SIZE = 32           # Number of images per training batch
SEED = 42                 # Random seed for reproducibility

# Paths relative to the project root (run this script from project root)
TRAIN_DIR = os.path.join("datasets", "train")
TEST_DIR = os.path.join("datasets", "test")


def load_dataset(directory: str, subset: str | None = None, validation_split: float = 0.0):
    """
    Load images from *directory* and return a batched tf.data.Dataset.

    Parameters
    ----------
    directory : str
        Path to the folder that contains one sub-folder per class.
    subset : str | None
        One of "training" or "validation" when *validation_split* > 0.
    validation_split : float
        Fraction of data to reserve for validation (only for training dir).

    Returns
    -------
    tf.data.Dataset
        Batched dataset with images scaled to [0, 1] and one-hot labels.
    """
    kwargs = dict(
        directory=directory,
        labels="inferred",          # Derive labels from sub-folder names
        label_mode="categorical",   # One-hot encode labels for multi-class output
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(subset == "training"),
        seed=SEED,
    )

    # Add split parameters only when a non-zero split is requested
    if validation_split > 0 and subset is not None:
        kwargs["validation_split"] = validation_split
        kwargs["subset"] = subset

    dataset = tf.keras.utils.image_dataset_from_directory(**kwargs)

    # Normalize pixel values from [0, 255] → [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(
        lambda images, labels: (normalization_layer(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch batches to overlap data loading with model computation
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_datasets():
    """
    Build and return the training, validation, and test datasets.

    A 20 % validation split is carved out from the training directory.
    The test directory is loaded separately without shuffling.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
        (train_ds, val_ds, test_ds)
    """
    print("Loading training dataset …")
    train_ds = load_dataset(TRAIN_DIR, subset="training", validation_split=0.2)

    print("Loading validation dataset …")
    val_ds = load_dataset(TRAIN_DIR, subset="validation", validation_split=0.2)

    print("Loading test dataset …")
    test_ds = load_dataset(TEST_DIR)

    # Report class names inferred from sub-folder names
    print(f"Class names: {train_ds.class_names}")
    print(
        f"Train batches: {len(train_ds)}  |  "
        f"Val batches: {len(val_ds)}  |  "
        f"Test batches: {len(test_ds)}"
    )

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    print("Preprocessing complete.")
