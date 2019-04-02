import time

import numpy as np
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.layers import (BatchNormalization, Concatenate, Conv2D, Dense,
                          Dropout, Flatten, GlobalMaxPooling2D, Input,
                          InputLayer)
from keras.losses import categorical_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import mobilenets
from model_utils.lr_schedule import lr_schedule
from data_utils.custom_augmentation import augment_image

batch_size = 32
img_shape = (320, 320, 1)
epochs = 500
seed = 2019
num_classes = 3

# train_datagen = ImageDataGenerator()
# train_generator = train_datagen.flow_from_directory(
#     directory="./data_test/train/",
#     target_size=img_shape[:-1],
#     color_mode="grayscale",
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=True,
#     seed=seed
# )

# test_datagen = ImageDataGenerator()
# test_generator = test_datagen.flow_from_directory(
#     directory="./data_test/test/",
#     target_size=img_shape[:-1],
#     color_mode="grayscale",
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=False,
#     seed=seed
# )

train_data_dir = '/Users/lamhoangtung/cds_data/final/flowable/raw/'

train_datagen = ImageDataGenerator(
    preprocessing_function=augment_image,
    validation_split=0.2)  # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_shape[:-1],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    seed=seed,
    subset='training')  # set as training data

test_generator = train_datagen.flow_from_directory(
    train_data_dir,  # same directory as training data
    target_size=img_shape[:-1],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    seed=seed,
    subset='validation')  # set as validation data

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = test_generator.n//test_generator.batch_size

# Model
model = mobilenets.MobileNet(
    input_shape=img_shape, classes=num_classes, attention_module='cbam_block')

model.compile(optimizer=Adam(), loss=categorical_crossentropy,
              metrics=['accuracy'])

print(model.summary())


tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weight_path = "model/classification-{epoch:03d}-{val_acc:.5f}.hdf5"

checkpoint = ModelCheckpoint(filepath=weight_path,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [tensorboard, checkpoint, lr_reducer, lr_scheduler]

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=test_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=callbacks)
