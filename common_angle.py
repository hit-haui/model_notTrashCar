import cv2
import os 
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
from imgaug import augmenters as iaa
import custom_augmentation as ciaa



batch_size = 64
img_shape = (240, 320, 2)
epochs = 1000
path = '/home/vicker/Downloads/recored_data'

augment_object = iaa.Sequential([
    iaa.Add((-20, 20)),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=0.03*255)),
    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.GaussianBlur(sigma=0.5),
        iaa.MotionBlur(angle=(0,360))
    ])),
    iaa.GammaContrast(gamma=(0.5, 1.44)),
    iaa.Sometimes(0.2, iaa.OneOf([
        iaa.FastSnowyLandscape(lightness_threshold=(0,150)),
#         iaa.Fog()
    ])),
    iaa.OneOf([
      iaa.Sometimes(0.8,ciaa.RandomShadow()),
      iaa.Sometimes(0.4,ciaa.RandomGravel()),
      iaa.Sometimes(0.2,ciaa.RandomSunFlare())
    ])
])

augment_min = iaa.Sequential([
    iaa.Sometimes(0.8,iaa.Add((-20, 20)))
])

def load_data(type_data):
    
    if type_data == 'train_generator':
        dataset = json.loads(open(path+'/over_sampled_label.json', 'r').read())
    elif type_data == 'val_generator':
        dataset = json.loads(open(path+'/test.json', 'r').read())
    img_rgb_path = []
    img_depth_path = []
    angles = []

    for each_sample in tqdm(dataset):
        rgb_path = each_sample['rgb_img_path']
        depth_path = each_sample['depth_img_path']
        img_rgb_path.append(rgb_path)
        img_depth_path.append(depth_path)
        angle = each_sample['angle'] + 60
        angles.append(angle)

    return img_rgb_path, img_depth_path, angles

def generator(type_data):
    """
    """
    
    img_rgb_path, img_depth_path, angles = load_data(type_data = type_data)
    img_rgb_path = np.array(img_rgb_path)
    img_depth_path = np.array(img_depth_path)
    angles = np.array(angles)
    order = np.arange(len(img_rgb_path))
    
    while True:

        # Shuffle training data
        np.random.shuffle(order)
        x = img_rgb_path[order]
        y1 = angles[order]
        x2 = img_depth_path[order]
        img_list = []
        angle_list = []

        for index in range(batch_size):
            count = 0
            file_name = x[index].split('/')[-1]
            
            for each in file_name:
                if each == '_':
                    count +=1
            img_rgb = cv2.imread(x[index])
            if count == 2 :
                img_rgb = augment_object.augment_image(img_rgb)
            else :
                img_rgb = augment_min.augment_image(img_rgb)

            img_rgb = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
            img_depth = cv2.imread(x2[index],0)
            
            img = np.expand_dims(img_rgb, axis=2)
            img_list.append(np.dstack((img_rgb,img_depth)))
            angle = y1[index]
            angle_list.append(angle) 
            

        img_list = np.array(img_list)
        angle_list = np.array(angle_list)
        
    
        yield (img_list), (angle_list)
            
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
