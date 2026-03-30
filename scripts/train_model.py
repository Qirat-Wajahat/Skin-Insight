"""
train_model.py
--------------
Builds, compiles, trains, and saves a CNN model for skin condition classification.

Skin classes (6):
    0 – Acne
    1 – Blackheads
    2 – Dark Spots
    3 – Normal
    4 – Pores
    5 – Wrinkles

Run from the project root:
    python scripts/train_model.py
"""

import os
import sys

# Ensure the scripts directory is on the Python path so preprocess can be imported
sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from preprocess import get_datasets

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224, 3)   # Height × Width × Channels
NUM_CLASSES = 6              # Acne, Blackheads, Dark Spots, Normal, Pores, Wrinkles
EPOCHS = 30                  # Training epochs (increase for better accuracy)
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = os.path.join("backend", "models", "skin_model.h5")


def build_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    Construct the CNN architecture.

    Architecture overview:
        Data Augmentation  → reduce overfitting via random flips/rotations
        Conv Block 1       → 32 filters, 3×3, ReLU + MaxPool
        Conv Block 2       → 64 filters, 3×3, ReLU + MaxPool
        Conv Block 3       → 128 filters, 3×3, ReLU + MaxPool
        Conv Block 4       → 256 filters, 3×3, ReLU + MaxPool
        GlobalAveragePool  → flatten feature maps into a vector
        Dense(256, ReLU)   → fully-connected feature extraction
        Dropout(0.5)       → regularisation to prevent overfitting
        Dense(num_classes, softmax) → probability distribution over classes

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input image, e.g. (224, 224, 3).
    num_classes : int
        Number of output classes.

    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            # ── Input layer ────────────────────────────────────────────────
            keras.Input(shape=input_shape),

            # ── Data augmentation (only active during training) ─────────────
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),

            # ── Convolutional Block 1: detect low-level features (edges) ───
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),    # Halve spatial dimensions

            # ── Convolutional Block 2: detect mid-level features ───────────
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # ── Convolutional Block 3: detect higher-level features ─────────
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # ── Convolutional Block 4: detect complex patterns ──────────────
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # ── Reduce spatial feature maps to a 1-D vector ────────────────
            layers.GlobalAveragePooling2D(),

            # ── Fully-connected classification head ────────────────────────
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),           # Drop 50% of neurons to reduce overfitting
            layers.Dense(num_classes, activation="softmax"),  # Output probabilities
        ],
        name="SkinInsight_CNN",
    )
    return model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    """
    Compile the model with Adam optimiser and categorical cross-entropy loss.

    Parameters
    ----------
    model : keras.Model
        Uncompiled Keras model.
    learning_rate : float
        Learning rate for the Adam optimiser.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",   # Multi-class one-hot labels
        metrics=["accuracy"],
    )


def plot_history(history: keras.callbacks.History) -> None:
    """
    Plot and save training / validation accuracy and loss curves.

    Parameters
    ----------
    history : keras.callbacks.History
        Object returned by model.fit().
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy curve ──────────────────────────────────────────────────────
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # ── Loss curve ──────────────────────────────────────────────────────────
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("backend", "models", "training_history.png"))
    print("Training history plot saved to backend/models/training_history.png")


def main():
    # ── 1. Load datasets ────────────────────────────────────────────────────
    train_ds, val_ds, _ = get_datasets()

    # ── 2. Build the CNN model ───────────────────────────────────────────────
    model = build_model(IMAGE_SIZE, NUM_CLASSES)
    model.summary()

    # ── 3. Compile the model ────────────────────────────────────────────────
    compile_model(model, LEARNING_RATE)

    # ── 4. Define callbacks ─────────────────────────────────────────────────
    callbacks = [
        # Stop training early if validation loss does not improve for 5 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        # Save the best model weights during training
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── 5. Train the model ──────────────────────────────────────────────────
    print("\nStarting training …")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # ── 6. Save the final model ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # ── 7. Plot training history ─────────────────────────────────────────────
    plot_history(history)


if __name__ == "__main__":
    main()
