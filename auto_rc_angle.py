import tensorflow as tf

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, Lambda
from keras.optimizers import Adam
from keras import regularizers

from config import *
from common_angle import *

img_size = (240, 320, 1)

batch_size = 64
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