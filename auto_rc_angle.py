import tensorflow as tf

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, Lambda
from keras.optimizers import Adam
from keras import regularizers

from config import *
from common_angle import *


# Data generator
train_generator = generator(type_data = 'train_generator') 
val_generator = generator(type_data = 'val_generator')

# Model
input_shape = Input(shape=img_shape, name='input_shape')

X = input_shape

# PilotNet five convolutional layers
X = Conv2D(filters=24, kernel_size=(5, 5),
           strides=(2, 2), activation='relu')(X)
X = Conv2D(filters=36, kernel_size=(5, 5),
           strides=(2, 2), activation='relu')(X)
X = Conv2D(filters=48, kernel_size=(5, 5),
           strides=(2, 2), activation='relu')(X)
X = Conv2D(filters=64, kernel_size=(3, 3),
           strides=(2, 2), activation='relu')(X)
X = Conv2D(filters=64, kernel_size=(3, 3),
           strides=(1, 1), activation='relu')(X)

# Flattened layer
X = Flatten()(X)

# PilotNet first fully-connected layer + dropout
X = Dense(units=200, activation='relu')(X)
X = Dropout(rate=0.1)(X)

# PilotNet second fully-connected layer + dropout
X = Dense(units=50, activation='relu')(X)
X = Dropout(rate=0.1)(X)

# PilotNet third fully-connected layer + dropout
X = Dense(units=10, activation='relu')(X)
X = Dropout(rate=0.1)(X)

# # PilotNet fourth fully-connected layer + dropout
# X = Dense(units=10, activation='relu')(X)
# X = Dropout(rate=0.1)(X)


steering = Dense(1, activation='relu', name='steering')(X)
#speed = Dense(1, activation='relu', name='speed')(X)



# Build and compile model
model = Model(inputs=[input_shape], outputs=[steering])

print(model.summary())

# Loss and optimizer
model.compile(optimizer = Adam(),
                loss = 'mse')


# Train the model
weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=batch_size, epochs=epochs,
                              validation_data=val_generator, validation_steps=batch_size,
                              callbacks=get_callback(weight_path=weight_path, batch_size=batch_size,))
