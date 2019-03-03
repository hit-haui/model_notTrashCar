from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model

from config import *
from common import *


epochs = 100
train_split = 0.8
val_split = 0.2

batch_size = 2

early_stop = False
test_overfit_single_batch = True

# Data generator
train_generator = train_generator(
    (160,120,3), batch_size,test_overfit_single_batch, train_split)

val_generator = val_generator(
    (160,120,3), batch_size,test_overfit_single_batch, val_split)
# Model
img_in = Input(shape=(120, 160, 4), name='img_in')
x = img_in
x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

x = Flatten(name='flattened')(x)
x = Dense(100, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(.1)(x)
# categorical output of the angle
angle_out = Dense(1, activation='relu', name='angle_out')(x)

# continous output of throttle
throttle_out = Dense(1, activation='relu', name='throttle_out')(x)

model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
# model summary
print(model.summary())


model.compile(optimizer='adam',
                loss={'angle_out': 'mean_squared_error',
                    'throttle_out': 'mean_squared_error'},
                loss_weights={'angle_out': 0.5, 'throttle_out': .5})


# Train the model
weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=batch_size, epochs=epochs,
                    validation_data=val_generator, validation_steps=batch_size,
                    callbacks=get_callback(weight_path=weight_path, batch_size=batch_size,early_stop = early_stop))
