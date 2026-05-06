"""
LeafLens — Model Training Script
Run this in Jupyter Notebook or directly via: python train_model.py
Dataset: PlantVillage (38 disease classes)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
NUM_CLASSES = 38
DATASET_DIR = "../dataset/PlantVillage"   # change if your path differs
MODEL_DIR   = "."

# ──────────────────────────────────────────────
# DATA GENERATORS  (augmentation for training)
# ──────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Save class indices so the Flask API can decode predictions
class_indices = train_generator.class_indices
class_names   = {v: k for k, v in class_indices.items()}   # {0: 'Apple_scab', ...}

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print(f"Classes found: {len(class_indices)}")
print(f"Training samples:   {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# ──────────────────────────────────────────────
# MODEL — MobileNetV2 + custom classification head
# ──────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False   # freeze pretrained weights initially

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ──────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "leaflens_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(log_dir="./logs", histogram_freq=1)
]

# ──────────────────────────────────────────────
# PHASE 1 — Train head only (frozen base)
# ──────────────────────────────────────────────
print("\n=== PHASE 1: Training classification head ===")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# ──────────────────────────────────────────────
# PHASE 2 — Fine-tune top layers of MobileNetV2
# ──────────────────────────────────────────────
print("\n=== PHASE 2: Fine-tuning top MobileNetV2 layers ===")
# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ──────────────────────────────────────────────
# SAVE FINAL MODEL
# ──────────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, "leaflens_model.h5"))
print("Model saved → leaflens_model.h5")

# ──────────────────────────────────────────────
# PLOT TRAINING CURVES
# ──────────────────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    val  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    vloss= h1.history["val_loss"] + h2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(acc,  label="Train accuracy")
    ax1.plot(val,  label="Val accuracy")
    ax1.axvline(x=len(h1.history["accuracy"])-1, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2.plot(loss,  label="Train loss")
    ax2.plot(vloss, label="Val loss")
    ax2.axvline(x=len(h1.history["loss"])-1, color="gray", linestyle="--")
    ax2.set_title("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved training_curves.png")

plot_history(history1, history2)
