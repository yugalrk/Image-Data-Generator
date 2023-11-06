# Image-Data-Generator
An Image Generator that can augment existing images to be used for model training.

# This Image Generator relies on several Python libraries that you may need to install if you haven't already:#
.NumPy
.scikit-image (skimage)
.Matplotlib

# Configuration #
file_path: Path to the directory containing image data.
label_path: Path to a JSON file that maps image filenames to their corresponding labels.
batch_size: The number of images in each batch.
image_size: The desired image size for resizing.
rotation: Apply random image rotation (90, 180, 270 degrees) if set to True.
mirroring: Apply horizontal mirroring if set to True.
shuffle: Shuffle the dataset for each epoch if set to True.

# Data Augmentation #
The generator supports two types of data augmentation:

1. Rotation: Random rotation of images by 90, 180, or 270 degrees.
2. Mirroring: Horizontal flipping of images with a 50% probability.
These data augmentation techniques are useful for improving the variety of training data, which can lead to better model performance.

Methods:
next(): Retrieve the next batch of images and labels.
current_epoch(): Get the current epoch count.
class_name(label): Get the class name based on the label.
show(): Display the batch of images.

# Contributing: #
If you'd like to contribute to this project or report issues, please follow the standard open-source guidelines and create a pull request or issue on the GitHub repository.
