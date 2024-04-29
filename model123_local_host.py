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
        loss='sparse_categorical_crossentropy',
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
    plt.savefig('m2air.png')
    plt.show()

# Function to save the model
def save_model(model, filename):
    model.save(filename)

# Main function
def main():
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    CHANNELS = 3
    EPOCHS = 32

    train_dir = "train_ds"
    test_dir = "test_ds"
    val_dir = "validation_ds"

    train_dataset, test_dataset, val _dataset = load_datasets(train_dir, test_dir, val_dir, BATCH_SIZE, IMAGE_SIZE)
    n_classes = len(train_dataset.class_names)

    input_shape = ( BATCH_SIZE ,IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    model = create_compile_model(input_shape, n_classes)

    history,training_time = train_model(model, train_dataset, vali_dataset, EPOCHS)
    print("Trianing time is", training_time/60 ,"Mins")

    scores = evaluate_model(model, test_dataset)
    print("Loss is ",scores[0])
    print("Accuracy is ",scores[1])
    

    plot_training_history(history, EPOCHS)

    save_model(model, "my_model32.h5")
    save_model(model,"my_model32.hdf5")
    save_model(model, "my_model32.keras")
    
if __name__ == "__main__":
    main()

