import cv2
import json
import numpy as np 
import keras
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Input, Dropout, InputLayer, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.losses import categorical_crossentropy
import time
from tqdm import tqdm

path = '/home/vicker/Downloads/dataset_1552725149.9200957/'
dataset = json.loads(open(path+'train_traffic.json','r').read())
batch_size = 64
epochs = 2
img_shape = (240,640,1)
imgs = []
lables = []

for each_sample in tqdm(dataset):
    traffic = [0,0,0,0]
    img = cv2.imread(each_sample['traffic_img_path'],0)
    cv2.destroyAllWindows()
    lable = each_sample['status_traffic']
    img = np.expand_dims(img, axis=-1)
    traffic[lable] = 1
    imgs.append(img)
    lables.append(traffic)

imgs = np.array(imgs)
lables = np.array(lables)

x_train, x_test, y_train, y_test = train_test_split(imgs, lables, test_size=0.2, random_state=2019, shuffle=True)

#Model
model = Sequential()
model.add(InputLayer(input_shape=img_shape))
model.add(Conv2D(8, (5,5), strides = (3,3), activation = 'relu'))
model.add(Conv2D(16, (5,5), strides = (2,2), activation = 'relu'))
model.add(Conv2D(32, (5,5), strides = (2,2), activation = 'relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

print(model.summary())

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

#callback

tensorboard = TensorBoard(log_dir="logs/detect_traffic_{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weight_path = "model_traffic/detect_traffic-{epoch:03d}-{val_loss:.5f}.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

callbacks = [tensorboard, checkpoint]

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test), shuffle=True)

#Predict
lables = []
for filename in os.listdir(path+'rgb/'):
    img = cv2.imread(path+'rgb/'+filename,0)
    print('predict',filename)
    h,w = img.shape
    img = img[:int(h/2),:w]
    img = np.expand_dims(img, axis=-1)
    traffic_status = model.predict(np.array([img]))[0]
    if traffic_status[0] == float(1) :
        status = 'no traffic'
    elif traffic_status[1] == float(1):
        status = 'turn left'
    elif traffic_status[2] == float(1):
        status = 'stop'
    elif traffic_status[3] == float(1) :
        status = 'right'

    sample = {
        'rgb_img_path' : './rgb/'+filename,
        'status_traffic:' : status
    }
    lables.append(sample)

with open(os.path.join(path, 'abc.json'), 'w', encoding='utf-8') as outfile:
    json.dump(lables, outfile, ensure_ascii=False, sort_keys=False, indent=4)
    outfile.write("\n")




