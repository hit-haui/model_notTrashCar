import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Convolution2D, ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import optimizers
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import model_architecture
from generator import generator
# Load the all the samples from the csv file and store them into single sample lines	
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != 'center':
            samples.append(line)
print("Data sets loaded!")
# Hyper-parameter
epochs = 20
sample_size = 64
learning_rate = 1e-04
# Get training (80%) and validation (20%) sample lines
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# Init generators
train_generator = generator(train_samples, sample_size, 'train')
validation_generator = generator(valid_samples, sample_size, 'valid')



# Show message that we start
print("Training network..")
print()

# Init the model
model = model_architecture()

# Loss and optimizer
model.compile(loss='mse', optimizer=optimizers.Adam(lr=learning_rate))

# Callbacks for checkpoints and early stop
check_point = ModelCheckpoint('./checkpoints/model-e{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

# Train the model

history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3, validation_data=validation_generator,nb_val_samples=len(valid_samples), nb_epoch=epochs, verbose=1, callbacks=[early_stop, check_point])

# Save it to a file and show message again
model.save('model.h5')
print()
print("Network trained!")