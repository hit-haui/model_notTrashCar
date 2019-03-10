import cv2
import os 
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import time



batch_size = 64
img_shape = (240, 320, 1)
epochs = 1000
path = '/home/vicker/Downloads/recored_data/'

def create_generator(img, label, batch_size):
    generator = ImageDataGenerator()
    return generator.flow(x=img, y=label, batch_size=batch_size)

def load_data():
    
    dataset = json.loads(open(path+'/key_data.json', 'r').read())
    imgs = []
    angles = []
    for each_sample in tqdm(dataset):
        new_path = os.path.join(path, 'rgb', '{}_rgb.jpg'.format(each_sample['index']))
        # print(new_path)
        img = cv2.imread(new_path, 0)
        angle = each_sample['angle'] + 60
        img = np.expand_dims(img, axis=2)
        imgs.append(img)
        angles.append(angle)

    imgs = np.array(imgs)
    angles = np.array(angles)

    x_train, x_test, y_train, y_test = train_test_split(imgs, angles, test_size=0.2,random_states= 2019)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_train.shape)
    # import pdb; pdb.set_trace()

    train_gen = create_generator(x_train, y_train, batch_size)
    test_gen = create_generator(x_test, y_test, batch_size)
    return train_gen,test_gen

def get_callback(weight_path, batch_size):
    # Callbacks
    # earlystop = EarlyStopping(
    #     monitor='val_loss', patience=5, verbose=0, mode='min')

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                              batch_size=batch_size, write_images=True)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                               patience=5, min_lr=1e-5)
   
    callbacks = [tensorboard, checkpoint]
    return callbacks