from keras.layers import BatchNormalization, Conv2D, Dense, Flatten
from keras.models import Sequential


# Define the model architecture, in this case use the NVIDIA end-to-end driving model
def model_architecture():
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=1, input_shape=(66, 200, 3)))
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
    model.add(Dense(2, activation='relu'))
    return model