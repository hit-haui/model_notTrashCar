import cv2
import json
import numpy as np 
import keras
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, Input, Dropout, InputLayer, GlobalMaxPooling2D, Flatten, Concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import time
from tqdm import tqdm


batch_size = 32
img_shape = (320,320,1)
epochs = 700

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    directory="/home/vicker/Documents/train/",
    target_size=(320, 320),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=2019
)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    directory="/home/vicker/Documents/test/",
    target_size=(320, 320),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=2019
)

#Model
model = Sequential()

model.add(BatchNormalization(input_shape = img_shape))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

print(model.summary())

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = test_generator.n//test_generator.batch_size


tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                              batch_size=batch_size, write_images=True)


weight_path = "model/read_data_3class-{epoch:03d}-{val_loss:.5f}.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=0, save_best_only=False,
                                save_weights_only=False, mode='auto', period=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=5, min_lr=1e-5)

callbacks = [tensorboard, checkpoint]


# model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test), shuffle=True)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=test_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=callbacks)


