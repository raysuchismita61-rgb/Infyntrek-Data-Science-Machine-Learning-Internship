import cv2
import numpy as np


def preprocess_image(image_path, image_size=128):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (image_size, image_size)
    )

    image = image.astype(np.float32) / 255.0

    return image


def preprocess_mask(mask_path, image_size=128):
    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    mask = cv2.resize(
        mask,
        (image_size, image_size)
    )

    mask = mask.astype(np.float32) / 255.0

    mask = np.expand_dims(mask, axis=-1)

    return mask