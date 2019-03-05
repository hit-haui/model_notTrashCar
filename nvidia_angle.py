from keras import optimizers
from keras.layers import BatchNormalization, Conv2D, Flatten, Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from config import *
from common_angle import *


img_size = (200, 66, 3)
epochs = 200
learning_rate = 0.01
batch_size = 64

train_split = 0.8
val_split = 0.2
 
early_stop = False
test_overfit_single_batch = True


train_generator = train_generator(
    (200, 66 , 3), batch_size, test_overfit_single_batch, train_split)

val_generator = val_generator(
    (200, 66, 3), batch_size,test_overfit_single_batch, val_split)
# Init the model


model = Sequential()

model.add(BatchNormalization(epsilon=0.001, axis=1,input_shape=(66, 200, 3)))


model.add(Conv2D(24, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='relu',name='steering'))




print(model.summary())


# Loss and optimizer
model.compile(optimizer = Adam(lr = learning_rate),
                loss = 'mse')


# Train the model
weight_path = "model/first-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=batch_size, epochs=epochs,
                              validation_data=val_generator, validation_steps=batch_size,
                              callbacks=get_callback(weight_path=weight_path, batch_size=batch_size, early_stop = early_stop))
