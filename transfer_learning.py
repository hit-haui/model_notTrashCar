from keras import optimizers
from keras.layers import BatchNormalization, Conv2D, Flatten, Dense, Input, MaxPooling2D, GlobalMaxPooling2D, Concatenate
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.models import Model
from config import *
from common_angle import *
from keras.applications import ResNet50
import json
import cv2


train_generator = generator(type_data='train_generator')
val_generator = generator(type_data='val_generator')
# Init the model1

input_img = Input(shape=(img_shape), name='input_shape')
# X = input_img
T = Input(shape=([3, ]))

base_model = ResNet50(include_top=False, weights='imagenet',
                      input_shape=img_shape, input_tensor=input_img)
x = base_model.output

X = GlobalMaxPooling2D()(x)
concatenateXT = Concatenate(axis=1)([X, T])
X = Dense(1024, activation='relu')(concatenateXT)
X = Dense(512, activation='relu')(X)
X = Dense(64, activation='relu')(X)
X = Dense(1, activation='relu')(X)
model = Model(input=[input_img, T], output=[X])
for layer in model.layers[:-4]:
    layer.trainable = False
print(model.summary())


# Loss and optimizer
model.compile(optimizer=Adam(),
              loss='mse')

total_sample = len(json.loads(
    open(path+'/over_sampled_label.json', 'r').read()))

# Train the model
weight_path = "model/TL_read_data_2chanel-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=total_sample//batch_size, epochs=epochs,
                    validation_data=val_generator, validation_steps=batch_size,
                    callbacks=get_callback(weight_path=weight_path, batch_size=batch_size,))
