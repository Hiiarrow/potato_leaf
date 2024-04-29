import sys
import subprocess
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

def should_install_requirement(requirement):
    should_install = False
    try:
        pkg_resources.require(requirement)
    except (DistributionNotFound, VersionConflict):
        should_install = True
    return should_install


def install_packages(requirement_list):
    try:
        requirements = [
            requirement
            for requirement in requirement_list
            if should_install_requirement(requirement)
        ]
        if len(requirements) > 0:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])
        else:
            print("Requirements already satisfied.")

    except Exception as e:
        print(e)

requirement_list = ['gdown', 'numpy','matplotlib','tensorflow']
install_packages(requirement_list)

import os
import gdown
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras as keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import models, layers
from keras.models import Sequential
import time




# Function to load image datasets from directories
def load_datasets(train_dir, test_dir, val_dir, batch_size, image_size):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=(image_size, image_size),
    )
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=(image_size, image_size),
    )
    vali_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=(image_size, image_size),
    )
    return train_dataset, test_dataset, vali_dataset

# Function to apply data augmentation
def apply_data_augmentation():
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    return data_augmentation

# Function to create and compile the CNN model
def create_compile_model(input_shape, n_classes):
    model = models.Sequential([
        layers.Resizing(256,256),
        layers.Rescaling(1./255),
        apply_data_augmentation(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='nadam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# Function to train the model
def train_model(model, train_dataset, vali_dataset, epochs):
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=vali_dataset,
        epochs=epochs
    )
    end_time = time.time()
    training_time = end_time - start_time
    return history,training_time

# Function to evaluate the model
def evaluate_model(model, test_dataset):
    scores = model.evaluate(test_dataset)
    return scores

# Function to plot training history
def plot_training_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label='Training Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label='Training Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Function to save the model
def save_model(model, filename):
    model.save(filename)

DEFAULT_ROOT = os.getcwd()  # Current working directory
DATA_FOLDER_NAME = 'data'

def download_extract_data(file_ids, zip_file_names):
    extracted_folders = []
    
    for file_id, zip_file_name in zip(file_ids, zip_file_names):
        # Define the URL to download the file from Google Drive
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # Define the local directory to save the downloaded file
        root = os.path.join(DEFAULT_ROOT, DATA_FOLDER_NAME)
        
        # Create the root directory if it does not exist
        os.makedirs(root, exist_ok=True)
        
        # Define the local file path to save the downloaded file
        zip_file_path = os.path.join(root, zip_file_name)
        
        # Download the zip file from Google Drive if it doesn't exist locally
        if not os.path.exists(zip_file_path):
            gdown.download(url, zip_file_path, quiet=False)
        
        # Extract the downloaded zip file
        extracted_folder = os.path.splitext(zip_file_path)[0]
        if not os.path.exists(extracted_folder):
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(root)
        
        extracted_folders.append(extracted_folder)
    
    return extracted_folders
    

# Main function
def main():
   
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    CHANNELS = 3
    EPOCHS = 32

    file_ids = ['1Vw7OF95zc7eJRUZi9HKIgtt9bpk0jHrC', '1Y2kxv1vCRsw6eW7ZGOIx4_7tw6Hbrllc', '13GUOpGDzItUJKWsdCkMI4OGOIi0m-BAq']
    zip_file_names = ['test_ds.zip', 'train_ds.zip', 'validation_ds.zip']
    extracted_folders = download_extract_data(file_ids, zip_file_names)
    test_ds_path, train_ds_path, vali_ds_path = extracted_folders
    
    train_dir = train_ds_path
    test_dir = test_ds_path
    val_dir = vali_ds_path

    train_dataset, test_dataset, vali_dataset = load_datasets(train_dir, test_dir, val_dir, BATCH_SIZE, IMAGE_SIZE)
    n_classes = len(train_dataset.class_names)

    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    model = create_compile_model(input_shape, n_classes)

    history = train_model(model, train_dataset, vali_dataset, EPOCHS)
    print("Trianing time is", training_time/60 ,"Mins")

    scores = evaluate_model(model, test_dataset)
    print(scores)

    plot_training_history(history, EPOCHS)

    save_model(model, "my_model32.h5")

if __name__ == "__main__":
    main()

