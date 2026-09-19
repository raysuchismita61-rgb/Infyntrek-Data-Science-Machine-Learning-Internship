# Disaster Image Segmentation

## Project Overview

The Disaster Image Segmentation project is a Deep Learning and Computer Vision project developed as part of my **Infyntrek Data Science & Machine Learning Internship**.
The project uses image segmentation techniques to identify and segment disaster-related regions in images. A **U-Net-based deep learning architecture** is used to perform semantic segmentation of disaster images.

##  Objectives
- Preprocess disaster images and their corresponding segmentation masks.
- Train a deep learning model for image segmentation.
- Identify disaster-affected regions at the pixel level.
- Evaluate segmentation performance.
- Visualize the predicted segmentation masks.
- Generate segmentation results for sample images.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Deep Learning
- Computer Vision
- U-Net

## Dataset

The project uses disaster images together with corresponding segmentation masks.
Each image has a corresponding mask that indicates the regions of interest for segmentation.
Due to dataset size and repository file-size limitations, the complete image dataset and masks are not included in this GitHub repository.

## Project Workflow

1. Load disaster images and segmentation masks.
2. Resize images and masks to a fixed size.
3. Normalize image pixel values.
4. Prepare training and validation datasets.
5. Build the U-Net segmentation model.
6. Train the model using image-mask pairs.
7. Evaluate the model on validation/test data.
8. Generate predicted segmentation masks.
9. Visualize original images, ground-truth masks, and predicted masks.

## Model Architecture

The project uses a **U-Net** architecture for semantic image segmentation.
The U-Net consists of:

- Encoder (contracting path)
- Bottleneck
- Decoder (expanding path)
- Skip connections
- Pixel-level segmentation output

## Project Structure

```text
Project-3-Disaster-Image-Segmentation/
│
├── README.md
│
├── data/
│   ├── images/
│   └── masks/
│
├── src/
│   └── train_unet.py
│
└── outputs/
    ├── training_history.png
    └── segmentation_result.png
