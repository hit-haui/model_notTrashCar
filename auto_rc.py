import tensorflow as tf

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, Lambda
from keras.optimizers import Adam
from keras import regularizers

from config import *
from common import *

img_size = (120, 160, 4)

batch_size = 2
epochs = 100
learning_rate = 0.01

train_split = 0.8
val_split = 0.2
early_stop = False
test_overfit_single_batch = True

# Data generator
train_generator = train_generator(
    (160,120,3), batch_size,test_overfit_single_batch, train_split)

val_generator = val_generator(
    (160,120,3), batch_size,test_overfit_single_batch, val_split)

input_shape = Input(shape=img_size, name='input_shape')

X = input_shape

# PilotNet five convolutional layers
X = Conv2D(filters=24, kernel_size=(5, 5),
           strides=(2, 2), activation='relu')(X)
X = Conv2D(filters=36, kernel_size=(5, 5),
           strides=(2, 2), activation='relu')(X)
X = Conv2D(filters=48, kernel_size=(5, 5),
           strides=(2, 2), activation='relu')(X)
X = Conv2D(filters=64, kernel_size=(3, 3),
           strides=(1, 1), activation='relu')(X)
X = Conv2D(filters=64, kernel_size=(3, 3),
           strides=(1, 1), activation='relu')(X)

# Flattened layer
X = Flatten()(X)

# PilotNet first fully-connected layer + dropout
X = Dense(units=1152, activation='relu')(X)
X = Dropout(rate=0.1)(X)

# PilotNet second fully-connected layer + dropout
X = Dense(units=100, activation='relu')(X)
X = Dropout(rate=0.1)(X)

# PilotNet third fully-connected layer + dropout
X = Dense(units=50, activation='relu')(X)
X = Dropout(rate=0.1)(X)

# PilotNet fourth fully-connected layer + dropout
X = Dense(units=10, activation='relu')(X)
X = Dropout(rate=0.1)(X)


steering = Dense(1, activation='relu', name='steering')(X)
speed = Dense(1, activation='relu', name='speed')(X)



# Build and compile model
model = Model(inputs=[input_shape], outputs=[steering, speed])

print(model.summary())

# Compile model for linear
model.compile(optimizer=Adam(lr=learning_rate),
              loss={'steering': 'mse', 'speed': 'mse'})

# Train the model
weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=batch_size, epochs=epochs,
                    validation_data=val_generator, validation_steps=batch_size,
                    callbacks=get_callback(weight_path=weight_path, batch_size=batch_size,early_stop = early_stop))
