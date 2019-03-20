from keras import optimizers
from keras.layers import BatchNormalization, Conv2D, Flatten, Dense, Input, MaxPooling2D, GlobalMaxPooling2D, Concatenate
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.models import Model
from config import *
from common_angle import *
import json
import cv2
train_generator = generator(type_data = 'train_generator') 
val_generator = generator(type_data = 'val_generator')
# Init the model1

input_shape = Input(shape=(img_shape), name='input_shape')
X = input_shape
T = Input(shape=([3,]))

X = (BatchNormalization(input_shape=img_shape))(X)
X = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(X)
X = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(X)
X = BatchNormalization()(X)
X = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(X)
X = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(X)
X = BatchNormalization()(X)
X = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(X)
X = GlobalMaxPooling2D()(X)
#X = Flatten()(X)
concatenateXT = Concatenate(axis=1)([X,T])
X = Dense(1024, activation='relu')(concatenateXT)
X = Dense(512, activation='relu')(X)
X = Dense(64, activation='relu')(X)
X = Dense(1, activation='relu')(X)

model = Model(input = [input_shape, T], output = [X])
print(model.summary())


# Loss and optimizer
model.compile(optimizer = Adam(),
                loss = 'mse')

total_sample = len(json.loads(open(path+'/over_sampled_label.json', 'r').read()))

# Train the model
weight_path = "model/read_data_2chanel-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=total_sample//batch_size, epochs=epochs,
                              validation_data=val_generator, validation_steps=batch_size,
                              callbacks=get_callback(weight_path=weight_path, batch_size=batch_size,))
