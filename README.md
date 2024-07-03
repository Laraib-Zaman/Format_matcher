# Image Similarity Detection with InceptionV3

This project focuses on detecting and visualizing similar images using the InceptionV3 model and Structural Similarity Index (SSIM). The process involves unzipping a dataset of images, preprocessing them, and comparing them with a target image to identify similar images based on a defined threshold.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Convert PDFs to Images](#convert-pdfs-to-images)
   - [Unzip Files](#unzip-files)
   - [Resize Images](#resize-images)
   - [Detect Similar Images](#detect-similar-images)
4. [Functions](#functions)
5. [Examples](#examples)
6. [License](#license)

## Prerequisites

- Python 3.x
- Google Colab (or a local Jupyter environment)
- Internet connection (for downloading model weights and dependencies)

## Installation

Install the necessary libraries using the following commands:

```sh
!apt-get install poppler-utils
!pip install pdf2image
!pip install tensorflow scikit-image opencv-python matplotlib pandas numpy
```

## Usage

### Convert PDFs to Images

This function converts all PDFs in a specified directory to images.

```python
def pdf_to_image(path=''):
    """
    Convert PDFs to images and save them in the 'images' directory.
    
    Args:
        path (str): The directory containing PDF files.

    Returns:
        str: The path to the directory containing the converted images.
    """
```

### Unzip Files

This function unzips the provided file and extracts its contents.

```python
def unzipper(path, dest_path=None):
    """
    Unzip files and extract contents to the specified directory.
    
    Args:
        path (str): The source zip file path.
        dest_path (str, optional): The destination path to extract the zip file. Defaults to the current directory.

    Returns:
        str: The path to the directory containing the extracted images.
    """
```

### Resize Images

This function resizes all images in a directory to a fixed size.

```python
def img_resizer(source_path):
    """
    Resize images to 299x399 and return them as a list of numpy arrays.
    
    Args:
        source_path (str): The directory containing images to be resized.

    Returns:
        list: A list of resized images as numpy arrays.
        list: A list of image filenames corresponding to the resized images.
    """
```

### Detect Similar Images

This function detects images similar to a given target image based on a threshold.

```python
def incept(img_path, thres, zip_path):
    """
    Detect and visualize similar images using InceptionV3 and SSIM.
    
    Args:
        img_path (str): The path to the target image.
        thres (float): The similarity threshold for SSIM scores.
        zip_path (str): The path to the zip file containing the dataset images.

    Returns:
        list: Indices of images similar to the target image based on the threshold.
        list: Resized images from the dataset.
        list: SSIM scores for the dataset images compared to the target image.
        list or str: A list of tuples containing (image name, image) for similar images, or a message indicating no similar images were found.
    """
```

## Functions

- `pdf_to_image(path)`: Converts PDFs to images.
    - **Args:**
        - `path (str)`: The directory containing PDF files.
    - **Returns:**
        - `str`: The path to the directory containing the converted images.

- `unzipper(path, dest_path)`: Unzips files and extracts contents to a directory.
    - **Args:**
        - `path (str)`: The source zip file path.
        - `dest_path (str, optional)`: The destination path to extract the zip file. Defaults to the current directory.
    - **Returns:**
        - `str`: The path to the directory containing the extracted images.

- `img_resizer(source_path)`: Resizes images to a fixed size.
    - **Args:**
        - `source_path (str)`: The directory containing images to be resized.
    - **Returns:**
        - `list`: A list of resized images as numpy arrays.
        - `list`: A list of image filenames corresponding to the resized images.

- `incept(img_path, thres, zip_path)`: Detects similar images using InceptionV3 and SSIM.
    - **Args:**
        - `img_path (str)`: The path to the target image.
        - `thres (float)`: The similarity threshold for SSIM scores.
        - `zip_path (str)`: The path to the zip file containing the dataset images.
    - **Returns:**
        - `list`: Indices of images similar to the target image based on the threshold.
        - `list`: Resized images from the dataset.
        - `list`: SSIM scores for the dataset images compared to the target image.
        - `list or str`: A list of tuples containing (image name, image) for similar images, or a message indicating no similar images were found.

## Examples

1. **Convert PDFs to Images separately: **

```python
images_dir = pdf_to_image('/path/to/pdf/directory')
print(f"Images saved to {images_dir}")
```

2. **Unzip Files separately: **

```python
images_path = unzipper('/path/to/zip/file.zip', '/destination/directory')
print(f"Images extracted to {images_path}")
```

3. **Resize Images separately: **

```python
all_images, all_names = img_resizer('/path/to/images/directory')
print(f"Resized {len(all_images)} images")
```

4. **Detect Similar Images, from an image and a zipfile **

```python 
# you can change the threshold from 0.60 to any other float ranging from 0-1
top_indexes, imgs, ssim, cluster = incept('/path/to/target/image.jpg', 0.60, '/path/to/zip/file.zip')
if cluster != "no similar images":
    print(f"Found {len(cluster)} similar images")
else:
    print(cluster)
```
