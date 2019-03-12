import pandas as pd
import time
import cv2
import os
import numpy as np
from config import *
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau)
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_data(img_size, test_overfit_single_batch):
    input_data = pd.read_csv(train_csv)

    images = []
    speed_labels = []
    angle_labels = []
    count = 0
    for _, each_row in tqdm(input_data.iterrows(), desc="Preprocess data", total=input_data.shape[0]):
        # for index, each_row in tqdm(input_data.iterrows():
        speed = each_row['speed']
        angle = each_row['steering'] + 60
        image_path = each_row['center']
        img = cv2.imread(os.path.join('/home/vicker/Documents/end_to_end/data/', image_path))
        img = cv2.resize(img, img_size[:-1])
        img_depth = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        images.append(np.dstack((img,img_depth)))
        speed_labels.append(speed)
        angle_labels.append(angle)
        if test_overfit_single_batch == True:
            count += 1
        if count == 121:
            break

    images = np.array(images)
    speed_labels = np.array(speed_labels)
    angle_labels = np.array(angle_labels)
    return images, angle_labels, speed_labels


def split_train_data(img_size,test_overfit_single_batch, split_percentage=0.8):
    """

    """
    images_list, labels1_list, labels2_list = load_data(img_size, test_overfit_single_batch)

    # Split data to training set and validation set with 80:20 percentage
    train_x = np.array(images_list[:int(len(images_list) * split_percentage)])
    train_y1 = np.array(
        labels1_list[:int(len(images_list) * split_percentage)])
    train_y2 = np.array(
        labels2_list[:int(len(images_list) * split_percentage)])

    return train_x, train_y1, train_y2


def split_val_data(img_size,test_overfit_single_batch, split_percentage=0.2):
    """

    """
    images_list, labels1_list, labels2_list = load_data(img_size, test_overfit_single_batch)

    # Split data to training set and validation set with 80:20 percentage
    val_x = np.array(images_list[-int(len(images_list) * split_percentage):])
    val_y1 = np.array(labels1_list[-int(len(images_list) * split_percentage):])
    val_y2 = np.array(labels2_list[-int(len(images_list) * split_percentage):])

    return val_x, val_y1, val_y2


def train_generator(img_size, batch_size,test_overfit_single_batch, split_percentage=0.8):
    """

    """

    train_x, train_y1, train_y2 = split_train_data(
        img_size,test_overfit_single_batch, split_percentage)

    order = np.arange(len(train_x))

    while True:

        # Shuffle training data
        np.random.shuffle(order)
        x = train_x[order]
        y1 = train_y1[order]
        y2 = train_y2[order]

        for index in range(batch_size):
            x_train = x[index * batch_size:(index + 1) * batch_size]
            y1_train = y1[index * batch_size:(index + 1) * batch_size]
            y2_train = y2[index * batch_size:(index + 1) * batch_size]

            yield (x_train), [(y1_train), (y2_train)]


def val_generator(img_size, batch_size,test_overfit_single_batch, split_percentage=0.2):
    """

    """

    val_x, val_y1, val_y2 = split_val_data(
        img_size,test_overfit_single_batch, split_percentage)

    while True:
        # We don't shuffle validation data
        for index in range(batch_size):
            x_val = val_x[index * batch_size:(index + 1) * batch_size]
            y1_val = val_y1[index * batch_size:(index + 1) * batch_size]
            y2_val = val_y2[index * batch_size:(index + 1) * batch_size]

            yield (x_val), [(y1_val), (y2_val)]


def get_callback(weight_path, batch_size, early_stop):
    # Callbacks
    earlystop = EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='min')

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                              batch_size=batch_size, write_images=True)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=1e-5)
    if (early_stop == True):
        callbacks = [earlystop, tensorboard, checkpoint, reduce_lr]
    else: 
        callbacks = [tensorboard, checkpoint, reduce_lr]


    return callbacks
