import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import IMAGE_SIZE


def visualize_predictions(
    model_path,
    images,
    masks,
    output_dir,
    num_samples=5
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    predictions = model.predict(
        images[:num_samples]
    )

    for i in range(num_samples):

        prediction = predictions[i]

        prediction = (
            prediction > 0.5
        ).astype(np.float32)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(
            masks[i].squeeze(),
            cmap="gray"
        )
        plt.title("Actual Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(
            prediction.squeeze(),
            cmap="gray"
        )
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"{output_dir}/prediction_{i}.png"
        )

        plt.close()