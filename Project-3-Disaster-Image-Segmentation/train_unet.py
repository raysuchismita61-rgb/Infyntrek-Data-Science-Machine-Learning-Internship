import os
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    IMAGE_DIR,
    MASK_DIR,
    MODEL_DIR
)

from dataset import load_dataset
from unet import build_unet
from metrics import dice_coefficient, iou_score


print("Loading dataset...")

images, masks = load_dataset(
    IMAGE_DIR,
    MASK_DIR,
    IMAGE_SIZE
)

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)


X_train, X_test, y_train, y_test = train_test_split(
    images,
    masks,
    test_size=0.2,
    random_state=42
)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)


model = build_unet(
    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        dice_coefficient,
        iou_score
    ]
)

model.summary()


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/unet_best.keras",
    monitor="val_loss",
    save_best_only=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("Test results:", results)


print("Training completed!")