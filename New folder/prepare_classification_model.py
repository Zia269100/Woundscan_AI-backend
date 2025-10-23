# prepare_classification_model.py

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ------------------------------
# 1️⃣ Create models folder
# ------------------------------
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created folder: {MODEL_DIR}")

# ------------------------------
# 2️⃣ Build EfficientNetB0
# ------------------------------
try:
    print("Loading EfficientNetB0 with pre-trained ImageNet weights...")
    base_model = EfficientNetB0(
        weights="imagenet",          # pre-trained on ImageNet
        include_top=False,           # remove top layer
        input_shape=(224, 224, 3)    # RGB input
    )
except Exception as e:
    print(f"⚠️ Failed to load pre-trained weights: {e}")
    print("Loading model with random weights instead...")
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

# ------------------------------
# 3️⃣ Add classification head
# ------------------------------
x = GlobalAveragePooling2D()(base_model.output)
outputs = Dense(4, activation="softmax")(x)  # 4 classes: Cut, Burn, Ulcer, Infection

model = Model(inputs=base_model.input, outputs=outputs)

# Freeze base model for inference (optional, can train later)
base_model.trainable = False

# ------------------------------
# 4️⃣ Save model
# ------------------------------
model_path = os.path.join(MODEL_DIR, "classification_model.h5")
model.save(model_path)
print(f"✅ Classification model saved at {model_path}")
