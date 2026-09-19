import os
import numpy as np

from preprocessing import preprocess_image, preprocess_mask


def load_dataset(image_dir, mask_dir, image_size=128):

    images = []
    masks = []

    image_files = sorted(
        [
            filename
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(".jpg")
        ]
    )

    for filename in image_files:

        image_path = os.path.join(
            image_dir,
            filename
        )

        # Convert image filename (.jpg) to mask filename (.png)
        mask_filename = os.path.splitext(filename)[0] + ".png"

        mask_path = os.path.join(
            mask_dir,
            mask_filename
        )

        if not os.path.exists(mask_path):
            print(f"Mask not found for: {filename}")
            continue

        try:

            image = preprocess_image(
                image_path,
                image_size
            )

            mask = preprocess_mask(
                mask_path,
                image_size
            )

            images.append(image)
            masks.append(mask)

        except Exception as error:

            print(
                f"Error processing {filename}: {error}"
            )

    return (
        np.array(images),
        np.array(masks)
    )