import os
import cv2
import matplotlib.pyplot as plt


IMAGE_DIR = "data/images"

image_files = os.listdir(IMAGE_DIR)

print("Number of images:", len(image_files))


widths = []
heights = []

for filename in image_files:

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is not None:

        height, width = image.shape[:2]

        widths.append(width)
        heights.append(height)


print("Average width:", sum(widths) / len(widths))
print("Average height:", sum(heights) / len(heights))


plt.hist(widths)

plt.title("Image Width Distribution")
plt.xlabel("Width")
plt.ylabel("Number of Images")

plt.savefig(
    "outputs/plots/image_width_distribution.png"
)

plt.close()