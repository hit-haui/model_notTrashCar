import cv2
import json
import numpy as np 
import keras
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, Input, Dropout, InputLayer, GlobalMaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import time
from tqdm import tqdm

epochs = 20
img_shape = (80, 80, 1)
batch_size = 64


train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    directory="/home/linus/model_notTrashCar/traffic_sign_data/train",
    target_size=(80, 80),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=2019
)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    directory="/home/linus/model_notTrashCar/traffic_sign_data/test",
    target_size=(80, 80),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=2019
)

#Model
model = Sequential()
model.add(Conv2D(16, (5, 5), strides=(1, 1),
                 activation='relu', input_shape=img_shape))
model.add(Conv2D(32, (3,3), strides = (1,1), activation = 'relu'))
model.add(Conv2D(64, (3,3), strides = (1,1), activation = 'relu'))
model.add(GlobalMaxPooling2D())
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

print(model.summary())

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

#callback

tensorboard = TensorBoard(log_dir="./logs/traffic_sign_{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weight_path = "./model_traffic_sign/traffic_sign_{epoch:03d}_{val_acc:.5f}.hdf5"

earlystop = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='min')

checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=0, save_best_only=False,
                             save_weights_only=False, mode='auto', period=1)

callbacks = [tensorboard, checkpoint, earlystop]

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = test_generator.n//test_generator.batch_size

# model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test), shuffle=True)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=test_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=callbacks)

model.save('./model_traffic_sign/model_traffic_sign_finalll.hdf5')



