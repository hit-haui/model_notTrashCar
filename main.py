import pandas as pd
from config import *
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Convolution2D, ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import optimizers
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from model import model_architecture

# from tqdm import tqdm

# Read input
input_data = pd.read_csv(train_csv)



images = []
labels = []



# for index, each_row in tqdm(input_data.iterrows(), desc="Preprocess data"):
for index, each_row in input_data.iterrows():
    speed = each_row['speed']
    angle = each_row['steering'] + 90 
    image_path = each_row['center']
    img = cv2.imread(os.path.join('../end_to_end/data/',image_path))
    img = cv2.resize(img, img_size[:-1])
    images.append(img)
    labels.append(np.array([speed, angle]))


images = np.array(images)
labels = np.array(labels)

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.1)

gen = ImageDataGenerator(
    zoom_range=0.2,
    brightness_range=(10,15),
)

train_generator = gen.flow(x_train, y_train, batch_size= batch_size)
val_generator = gen.flow(x_val, y_val, batch_size= batch_size)

# Show message that we start
print("Training network..")
print()

# Init the model
model = model_architecture()

# Loss and optimizer
model.compile(loss='mse', optimizer=optimizers.Adam(lr=learning_rate))

# Callbacks for checkpoints and early stop
check_point = ModelCheckpoint('./checkpoints/model-e{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

# Train the model

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                            batch_size=batch_size, write_images=True
                        )
            
checkpoint = ModelCheckpoint(weight_path,monitor='val_loss',verbose=0,save_best_only=False,
                                save_weights_only=False, mode='auto',period=1 )

callbacks = [tensorboard,checkpoint]



history = model.fit_generator(generator= train_generator, steps_per_epoch = train_generator.n//batch_size, epochs= epochs,
                                validation_data = val_generator, validation_steps= val_generator.n//batch_size,
                                callbacks=callbacks)





