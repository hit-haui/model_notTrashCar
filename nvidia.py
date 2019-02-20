from keras import optimizers
from keras.layers import BatchNormalization, Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from config import *
from common import *


img_size = (200, 66, 3)
epochs = 100
learning_rate = 0.01
batch_size = 2

seq_length = 0
train_split = 0.8
val_split = 0.2

input_shape = (66, 200, 3)
# Data generator
#train_generator, val_generator = get_data_loader(batch_size=batch_size, img_size=img_size)
train_generator = train_generator(batch_size, seq_length, train_split, seq_train = False)

val_generator = val_generator(batch_size, seq_length, val_split, seq_train = False)

# Init the model
model = Sequential()
model.add(BatchNormalization(epsilon=0.001, axis=1, input_shape= input_shape))
model.add(Conv2D(24, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu',name='steering'))
model.add(Dense(1, activation='relu', name='speed'))



print(model.summary())


# Loss and optimizer
model.compile(optimizer = Adam(lr = learning_rate),
                loss = {'steering':'mse', 'speed':'mse'})


# Train the model
weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.n//batch_size, epochs=epochs,
                              validation_data=val_generator, validation_steps=val_generator.n//batch_size,
                              callbacks=get_callback(weight_path=weight_path, batch_size=batch_size))
