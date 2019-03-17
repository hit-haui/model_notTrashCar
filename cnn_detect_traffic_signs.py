import cv2
import json
import numpy as np 
import keras
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Input, Dropout, InputLayer
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
from tqdm import tqdm

path = '/home/vicker/Downloads/dataset_1552725149.9200957/'
dataset = json.loads(open(path+'train_traffic.json','r').read())
batch_size = 64
epochs = 100
img_shape = (240,640)
imgs = []
lables = []
for each_sample in tqdm(dataset):
    img = cv2.imread(each_sample['traffic_img_path'])
    cv2.destroyAllWindows()
    lable = each_sample['status_traffic'] 
    imgs.append(img)
    lables.append(lable)

imgs = np.array(imgs)
lables = np.array(lables)

x_train, x_test, y_train, y_test = train_test_split(imgs, lables, test_size=0.2, random_state=2019, shuffle=True)
print(x_train.shape)
print(x_test.shape)
#Model
model = Sequential()
model.add(InputLayer(input_shape=img_shape))
model.add(Conv2D(8, (5,5), strides = (2,2), activation = 'relu'))
model.add(Conv2D(16, (5,5), strides = (2,2), activation = 'relu'))
model.add(Conv2D(32, (5,5), strides = (2,2), activation = 'relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

print(model.summary())

model.compile(optimizer=Adam(), loss='mse')

#callback

tensorboard = TensorBoard(log_dir="logs/detect_traffic_{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weight_path = "model_traffic/detect_traffic-{epoch:03d}-{val_loss:.5f}.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

callbacks = [tensorboard, checkpoint]

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test), shuffle=True)




