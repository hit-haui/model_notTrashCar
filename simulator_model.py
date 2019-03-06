import keras
import json
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
from tqdm import tqdm

# Data
path = './data/'
dataset = json.loads(open(path+'/key_data.json', 'r').read())
batch_size = 64
img_shape = (240, 320, 1)
epochs = 1000

imgs = []
angles = []
for each_sample in tqdm(dataset):
    new_path = os.path.join(path, 'rgb', '{}_rgb.jpg'.format(each_sample['index']))
    # print(new_path)
    img = cv2.imread(new_path, 0)
    angle = each_sample['angle'] + 60
    img = np.expand_dims(img, axis=2)
    imgs.append(img)
    angles.append(angle)

imgs = np.array(imgs)
angles = np.array(angles)

x_train, x_test, y_train, y_test = train_test_split(imgs, angles, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_train.shape)
# import pdb; pdb.set_trace()


def create_generator(img, label, batch_size):
    generator = ImageDataGenerator()
    return generator.flow(x=img, y=label, batch_size=batch_size)

train_gen = create_generator(x_train, y_train, batch_size)
test_gen = create_generator(x_test, y_test, batch_size)

# Model
model = Sequential()
model.add(BatchNormalization(input_shape=img_shape))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

print(model.summary())

model.compile(optimizer=Adam(), loss='mse')

# Callback

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=False,
                             save_weights_only=False, mode='auto', period=1)

callbacks = [tensorboard, checkpoint]

model.fit_generator(generator=train_gen, steps_per_epoch=batch_size, epochs=epochs, validation_data=test_gen, validation_steps=batch_size, callbacks=callbacks)
