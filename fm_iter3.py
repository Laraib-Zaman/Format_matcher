# imports:
import os
import glob
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from flask import Flask

import cv2
# from google.colab import drive
import zipfile

# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from sklearn.cluster import KMeans


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from skimage import metrics


# libraries installed to convert pdfs to images
# !apt-get install poppler-utils
# !pip install pdf2image

from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError


class ImageSimilarityDetector:
    def __init__(self):
        self.inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='max')

    def pdf_to_image(self, path=''):
        """
        Convert PDFs to images and save them in the 'images' directory.

        Args:
            path (str): The directory containing PDF files.

        Returns:
            str: The path to the directory containing the converted images.
        """
        pdf_path = path if path.endswith("/pdfs") else os.path.join(path, 'pdfs')
        images_dir = os.path.join(path.replace("/pdfs", ""), 'images')

        os.makedirs(images_dir, exist_ok=True)

        pdf_files = [os.path.join(pdf_path, filename) for filename in os.listdir(pdf_path) if filename.endswith('.pdf')]

        for pdf_file in pdf_files:
            try:
                images = convert_from_path(pdf_file)
                for i, image in enumerate(images):
                    fname = os.path.join(images_dir, os.path.basename(pdf_file).replace('.pdf', f'_image{i}.jpg'))
                    image.save(fname, "JPEG")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

        return images_dir

    def extractor(self, zip_ref, dest_path):
        """
        Extract files from a zip archive to the specified directory.

        Args:
            zip_ref (ZipFile): The zip file reference.
            dest_path (str): The destination path to extract the zip file.
        """
        os.makedirs(dest_path, exist_ok=True)
        zip_ref.extractall(dest_path)

    def handle_duplicates(self, file_path):
        """
        Handle duplicate files by appending a counter to the filename.

        Args:
            file_path (str): The original file path

        Returns:
            str: The new file path with a counter appended if duplicates exist.
        """
        base, extension = os.path.splitext(file_path)
        counter = 1
        new_file_path = f"{base}_{counter}{extension}"
        while os.path.exists(new_file_path):
            counter += 1
            new_file_path = f"{base}_{counter}{extension}"
        return new_file_path

    def process_directory(self, path):
        """
        Process a directory to check for PDFs or images and handle them accordingly.

        Args:
            path (str): The directory path.

        Returns:
            str: The path to the directory containing the images.
        """
        if any(filename.endswith('.pdf') for filename in os.listdir(path)):
            return self.pdf_to_image(path)
        elif any(filename.endswith(('.jpg', '.jpeg', '.png')) for filename in os.listdir(path)):
            return path
        else:
            raise ValueError("Directory contains unsupported file types.")

    def unzipper(self, path, dest_path=None):
        """
        Unzip files and extract contents to the specified directory if not already extracted.

        Args:
            path (str): The source zip file path.
            dest_path (str, optional): The destination path to extract the zip file. Defaults to the current directory.

        Returns:
            str: The path to the directory containing the extracted images.
        """
        if dest_path is None:
            dest_path = os.getcwd()

        # Create a directory name based on the zip file name (without extension)
        zip_name = os.path.splitext(os.path.basename(path))[0]
        extract_dir = os.path.join(dest_path, zip_name)

        if os.path.exists(extract_dir):
            print(f"Using previously extracted contents of {path}")
            return os.path.join(extract_dir, 'images')

        with zipfile.ZipFile(path, 'r') as zip_ref:
            contents = zip_ref.infolist()
            filename = contents[0].filename
            if filename.endswith('.pdf'):
                self.extractor(zip_ref, os.path.join(extract_dir, 'pdfs'))
                return self.pdf_to_image(os.path.join(extract_dir, 'pdfs'))
            elif filename.endswith(('.jpg', '.jpeg', '.png')):
                self.extractor(zip_ref, os.path.join(extract_dir, 'images'))
                return os.path.join(extract_dir, 'images')
            else:
                raise ValueError("Unsupported file type in zip archive.")

    def img_resizer(self, source_path):
        """
        Resize images to 299x399 and return them as a list of numpy arrays.

        Args:
            source_path (str): The directory containing images to be resized.

        Returns:
            list: A list of resized images as numpy arrays.
            list: A list of image filenames corresponding to the resized images.
        """
        all_images = []
        all_names = []

        for img_name in os.listdir(source_path):
            img_path = os.path.join(source_path, img_name)
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
                image = cv2.resize(img, (299, 399))
                all_images.append(image)
                all_names.append(img_name)

        return all_images, all_names

    def no_of_item(self, path):
        """
        Helper function to count the number of items in a directory.

        Args:
            path (str): The directory path.

        Returns:
            int: Number of items in the directory.
        """
        return len([item for item in os.listdir(path) if os.path.isfile(os.path.join(path, item))])

    def incept(self, img_path, thres, input_path):
        """
        Detect and visualize similar images using InceptionV3 and SSIM.

        Args:
            img_path (str): The path to the target image.
            thres (float): The similarity threshold for SSIM scores.
            input_path (str): The path to the zip file or directory containing the dataset images.

        Returns:
            list: Indices of images similar to the target image based on the threshold.
            list: Resized images from the dataset.
            list: SSIM scores for the dataset images compared to the target image.
            list or str: A list of tuples containing (image name, image) for similar images, or a message indicating no similar images were found.
        """
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return [], [], [], "Error loading target image"

        new_img = np.array(pil_img)
        new_img = new_img.astype(np.uint8) if new_img.dtype != np.uint8 else new_img

        try:
            new_img_resized = cv2.resize(new_img, (299, 399))
        except Exception as e:
            print(f"Error resizing image {img_path}: {e}")
            return [], [], [], "Error resizing target image"

        if zipfile.is_zipfile(input_path):
            images_path = self.unzipper(input_path, os.path.dirname(input_path))
            print("images path in first itme use ", images_path )
        elif os.path.isdir(input_path):
            images_path = self.process_directory(input_path)
            print("images path used previously ", images_path )
        else:
            raise ValueError("Input path is neither a zip file nor a directory.")

        all_images, all_names = self.img_resizer(images_path)

        new_image_inception = inception_preprocess_input(np.array(new_img_resized))
        all_images_inception = inception_preprocess_input(np.array(all_images))

        new_features_inception = self.inception_model.predict(np.expand_dims(new_image_inception, axis=0))
        all_features_inception = self.inception_model.predict(all_images_inception)

        inception_ssim_scores = [
            metrics.structural_similarity(new_features_inception.flatten(), features_inception.flatten())
            for features_inception in all_features_inception
        ]

        threshold = thres
        inception_top_indices = [i for i, score in enumerate(inception_ssim_scores) if score >= threshold]

        if len(inception_top_indices) > 0:
            clustered_img = []
            plt.figure(figsize=(10, 10))
            for i, index in enumerate(inception_top_indices):
                plt.imshow(all_images[index])
                image_name = all_names[index]
                img = cv2.imread(os.path.join(images_path, image_name))
                if img is not None:
                    clustered_img.append((image_name, img))
                else:
                    print(f"Failed to read image {image_name} at index {index}")
                plt.title(f'Index: {index}, SSIM: {inception_ssim_scores[index]:.2f}, Name: {all_names[index]} ')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            return inception_top_indices, all_images, inception_ssim_scores, clustered_img
        else:
            return inception_top_indices, all_images, inception_ssim_scores, "no similar images"
