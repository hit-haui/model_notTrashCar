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
img_shape = (240, 320, 2)
epochs = 1000
path = '/home/hai/Downloads/recored_data'

def create_generator(img, label, batch_size):
    generator = ImageDataGenerator()
    return generator.flow(x=img, y=label, batch_size=batch_size)

def load_data():
    
    dataset = json.loads(open(path+'/over_sampled_label.json', 'r').read())
    imgs = []
    angles = []
    for each_sample in tqdm(dataset):
        rgb_path = each_sample['rgb_img_path']
        depth_path = each_sample['depth_img_path']
        # print(new_path)
        img_rgb = cv2.imread(rgb_path, 0)
        img_depth = cv2.imread(depth_path,0)
        angle = each_sample['angle'] + 60
        img = np.expand_dims(img_rgb, axis=2)
        imgs.append(np.dstack((img_rgb,img_depth)))
        angles.append(angle)

    imgs = np.array(imgs)
    angles = np.array(angles)

    x_train, x_test, y_train, y_test = train_test_split(imgs, angles, test_size=0.2)
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
