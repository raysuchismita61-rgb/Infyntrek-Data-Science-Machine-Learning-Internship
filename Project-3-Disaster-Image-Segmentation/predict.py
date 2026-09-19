import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import IMAGE_SIZE, IMAGE_DIR, MASK_DIR
from preprocessing import preprocess_image, preprocess_mask


# Load trained U-Net model
model = tf.keras.models.load_model(
    "models/unet_best.keras",
    compile=False
)


# Select an image
image_filename = sorted(
    [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(".jpg")
    ]
)[0]

image_path = os.path.join(IMAGE_DIR, image_filename)

# Matching mask
mask_filename = os.path.splitext(image_filename)[0] + ".png"
mask_path = os.path.join(MASK_DIR, mask_filename)


# Preprocess image and mask
image = preprocess_image(
    image_path,
    IMAGE_SIZE
)

mask = preprocess_mask(
    mask_path,
    IMAGE_SIZE
)


# Make prediction
prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)[0]


# Convert prediction to binary mask
prediction = (prediction > 0.5).astype(np.float32)


# Create output folder
os.makedirs("outputs", exist_ok=True)


# Create visualization
plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Disaster Image")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze(), cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(prediction.squeeze(), cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/segmentation_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("segmentation_result.png created successfully!")