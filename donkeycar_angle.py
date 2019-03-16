from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam

from config import *
from common_angle import *


# Data generator
train_generator = generator(type_data = 'train_generator') 
val_generator = generator(type_data = 'val_generator')


# Model
img_in = Input(shape=img_shape, name='img_input')
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


model = Model(inputs=img_in, outputs=angle_out)

# model summary
print(model.summary())


model.compile(optimizer = Adam(),
                loss = 'mse')
# Train the model
weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=batch_size, epochs=epochs,
                    validation_data=val_generator, validation_steps=batch_size,
                    callbacks=get_callback(weight_path=weight_path, batch_size=batch_size))
