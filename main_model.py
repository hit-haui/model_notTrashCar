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
from model_utils.clr import LRFinder

batch_size = 5
img_shape = (320, 320, 1)
epochs = 2
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

train_data_dir = '/home/linus/cds_data/train'

train_datagen = ImageDataGenerator(
    brightness_range=[0.5, 1.5],
    # preprocessing_function=augment_image,
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
# model = mobilenets.MobileNet(
#     input_shape=img_shape, classes=num_classes, attention_module='cbam_block')
model = mobilenets.MiniMobileNetV2(input_shape=img_shape, classes=num_classes)

model.compile(optimizer=Adam(), loss=categorical_crossentropy,
              metrics=['accuracy'])

print(model.summary())

tensorboard = TensorBoard(log_dir="./logs/cbam_mobile_net_newdata_{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weight_path = "./model/cbam_mini_mobilenetv2_{epoch:03d}_{val_acc:.5f}.hdf5"

checkpoint = ModelCheckpoint(filepath=weight_path,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

lr_finder = LRFinder(num_samples=train_generator.n, batch_size=batch_size,
                       minimum_lr=1e-6, maximum_lr=1,
                       # validation_data=(X_val, Y_val),
                       lr_scale='exp', save_dir='./lr_logs/', verbose=True)

LRFinder.plot_schedule_from_file('./lr_logs/', clip_beginning=10, clip_endding=5)


callbacks = [tensorboard, checkpoint, lr_finder]

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=test_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=callbacks)

lr_finder.plot_schedule(clip_beginning=10, clip_endding=5)
