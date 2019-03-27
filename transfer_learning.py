import json

from keras.applications import ResNet50
from keras.layers import (Concatenate, Dense, GlobalMaxPooling2D, Input)
from keras.models import Model
from keras.optimizers import Adam

from common_angle_tflearning import *

# Param world
batch_size = 32
img_shape = (224, 224, 3)
epochs = 700
data_path = '/home/linus/Desktop/final/'
total_sample = len(json.loads(
    open(data_path+'/over_sampled_label.json', 'r').read()))


train_generator = generator(type_data='train_generator',
                            path=data_path, batch_size=batch_size, img_shape=img_shape)
val_generator = generator(type_data='val_generator',
                          path=data_path, batch_size=batch_size, img_shape=img_shape)

# Init the model
input_img = Input(shape=(img_shape), name='input_shape')
input_tf_sign = Input(shape=([3, ]), name='input_traffic_sign')
cnn_model = ResNet50(include_top=False, weights='imagenet',
                     input_shape=img_shape, input_tensor=input_img)
X = cnn_model.output
X = GlobalMaxPooling2D()(X)
concat = Concatenate(axis=1)([X, input_tf_sign])
X = Dense(1024, activation='relu')(concat)
X = Dense(512, activation='relu')(X)
X = Dense(64, activation='relu')(X)
X = Dense(1, activation='relu')(X)

model = Model(input=[input_img, input_tf_sign], output=[X])

for layer in model.layers[:-4]:
    layer.trainable = False

print(model.summary())


# Loss and optimizer
model.compile(optimizer=Adam(), loss='mse')


# Train the model
weight_path = "model/resnet50_3chanel-{epoch:03d}-{val_loss:.5f}.hdf5"
model.fit_generator(generator=train_generator, steps_per_epoch=total_sample//batch_size, epochs=epochs,
                    validation_data=val_generator, validation_steps=batch_size,
                    callbacks=get_callback(weight_path=weight_path, batch_size=batch_size,))
