import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

#first we the data collection and processing function
from SLR_data_image_processor import collect_and_process_images

def train_sign_language_model(base_images_path, model_save_name='sign_language_model.h5', epochs=10, batch_size=32):
    """
        Collects processed image data and trains a CNN model for sign language recognition.

        Args:
            base_images_path (str): The base directory where image subfolders are located.
            model_save_name (str): The filename for saving the trained model.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """