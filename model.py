
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Conv2D,BatchNormalization,Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import optimizers
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the model architecture, in this case use the NVIDIA end-to-end driving model
def model_architecture():    
    # Define the model, we use Xavier initializer on each layer
    model = Sequential()

    model.add(BatchNormalization(epsilon=0.001, axis=1,input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='tanh'))
    
    return model




