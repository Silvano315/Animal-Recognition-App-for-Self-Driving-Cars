from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.constants import BINARY_LABELS, LABELS, BATCH_SIZE, SEED

# Function to create two data generator: train and test/val
def data_gen():

    train_datagen =  ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        #rescale = 1./255,
        shear_range = 0.2,
        brightness_range=[0.1, 1.5],
        zoom_range=[0.3, 1.5],
        horizontal_flip = True,
        fill_mode = 'nearest'
    )

    test_datagen = ImageDataGenerator(
        rescale = 1./255
    )

    return train_datagen, test_datagen


#Function to visualize images after data generator
def viz_images_generator(data_gen, X):

    images = data_gen.flow(X, batch_size=1)
    plt.figure(figsize=(16,16))
    for i in range(1,17):
        plt.subplot(4,4,i)
        batch = next(images)
        image = batch[0].astype("uint8")
        plt.imshow(image)
        plt.axis("off")
    plt.show()


#Function to normalize features
def norm_X(X_train, X_test):

    train_norm = (X_train / 255.0).astype('float32')
    test_norm = (X_test / 255.0).astype('float32')

    print("="*50)
    print(f"Min and Max values for X train: {np.min(train_norm)}, {np.max(train_norm)}")
    print(f"Min and Max values for X train: {np.min(test_norm)}, {np.max(test_norm)}")
    print("="*50)

    return train_norm, test_norm


#Function to transform y labels to binary labels: vehicles (0) and animals (1)
def binary_y(y_train, y_test):

    y_train_binary = np.vectorize(BINARY_LABELS.get)(y_train.flatten())
    y_test_binary = np.vectorize(BINARY_LABELS.get)(y_test.flatten())

    print("="*70)
    print(f"First 10 binary labels for training set: \n{y_train_binary[:10]}\n{np.vectorize(LABELS.get)(y_train.flatten())[:10]}")
    print("="*70)
    print(f"First 10 binary labels for test set: \n{y_test_binary[:10]}\n{np.vectorize(LABELS.get)(y_test.flatten())[:10]}")
    print("="*70)

    return y_train_binary, y_test_binary


# Funtion for splitting Training set into train and validation

def train_splitting(X_train, y_train, test_size = 0.1):

    X_train_split, X_val, y_train_binary_split, y_val_binary = train_test_split(X_train, y_train, test_size=test_size, 
                                                                        shuffle=True, stratify=y_train)

    print("="*60)
    print("Training information:")
    print(f"X train percentage for training after splitting: {(X_train_split.shape[0] * 100 / X_train.shape[0]):.0f}%")
    print(f"X train shape after splitting: {X_train_split.shape}")
    print(f"y train shape after splitting: {y_train_binary_split.shape}")
    print("="*60)
    print("Validation information:")
    print(f"X train percentage for validation after splitting: {(X_val.shape[0] * 100 / X_train.shape[0]):.0f}%")
    print(f"X val shape after splitting: {X_val.shape}")
    print(f"y val shape after splitting: {y_val_binary.shape}")
    print("="*60)

    return X_train_split, X_val, y_train_binary_split, y_val_binary


# Function to apply data generator to datasets
def apply_data_gen(train_datagen, X_train, y_train, test_datagen, X_val, y_val):
    train_gen = train_datagen.flow(
        X_train,
        y_train,
        batch_size = BATCH_SIZE,
        seed = SEED
    )

    val_gen = test_datagen.flow(
        X_val,
        y_val,
        batch_size = BATCH_SIZE,
        seed = SEED
    )

    return train_gen, val_gen