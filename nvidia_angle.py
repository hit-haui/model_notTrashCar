from keras import optimizers
from keras.layers import BatchNormalization, Conv2D, Flatten, Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from config import *
from common_angle import *


train_generator = train_generator() 
val_generator = val_generator()

# Init the model
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


# Loss and optimizer
model.compile(optimizer = Adam(),
                loss = 'mse')


# Train the model
weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=batch_size, epochs=epochs,
                              validation_data=val_generator, validation_steps=batch_size,
                              callbacks=get_callback(weight_path=weight_path, batch_size=batch_size,))
