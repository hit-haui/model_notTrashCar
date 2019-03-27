import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from tqdm import tqdm

import custom_augmentation as ciaa

model_traffic = load_model(
    '/home/linus/model_notTrashCar/model_traffic_sign/detect2_traffic-016-0.98212.hdf5')
graph = tf.get_default_graph()

# augment_object = iaa.Sequential([
#     iaa.Add((-20, 20)),
#     iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=0.03*255)),
#     iaa.Sometimes(0.5, iaa.OneOf([
#         iaa.GaussianBlur(sigma=0.5),
#         iaa.MotionBlur(angle=(0, 360))
#     ])),
#     iaa.GammaContrast(gamma=(0.5, 1.44)),
#     iaa.Sometimes(0.2, iaa.OneOf([
#         iaa.FastSnowyLandscape(lightness_threshold=(0, 150)),
#         #         iaa.Fog()
#     ])),
#     iaa.OneOf([
#         iaa.Sometimes(0.8, ciaa.RandomShadow()),
#         iaa.Sometimes(0.4, ciaa.RandomGravel()),
#         iaa.Sometimes(0.2, ciaa.RandomSunFlare())
#     ])
# ])


augment_object = augment_min = iaa.Sequential([
    iaa.Sometimes(0.8, iaa.Add((-20, 20)))
])


def load_data(type_data, path):
    if type_data == 'train_generator':
        dataset = json.loads(open(path + '/over_sampled_label.json', 'r').read())
    elif type_data == 'val_generator':
        dataset = json.loads(open(path + '/test.json', 'r').read())
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


def get_predict(img):
    s = img.shape
    img = img[:s[0]//2, :]
    output = img.copy()
    raw = output.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 70, 70])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    color = res.copy()
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 17, 2)
    # detect circles in the image
    circles = cv2.HoughCircles(
        res, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=50)
    left = 0
    none = 0
    right = 0
    # ensure at least some circles were found
    if circles is not None and np.sum(circles) > 0:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # print('Got', len(circles), 'circles')

        # loop over the (x, y) coordinates and radius of the circles
        for index_phu, (x, y, r) in enumerate(circles):

            #print(x, y, r)
            # print(x, y, r)
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            # cv2.circle(output, (x, y), r, (0, 0, 255), 4)
            top_y = max(y - r - 10, 0)
            top_x = max(x - r - 10, 0)
            y_size = min(top_y+r*2+20, img.shape[0])
            x_size = min(top_x+r*2+20, img.shape[1])
            #print(top_x, top_y)
            # print(img.shape)
            img = img[top_y:y_size, top_x:x_size, :]
            h, w, c = img.shape
            if h and w != 0:
                if c != 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (80, 80))
                img = np.expand_dims(img, axis=-1)
                with graph.as_default():
                    traffic_list = model_traffic.predict(np.array([img]))[0]
                # print('predict:',traffic_list)
                l = traffic_list[0]
                n = traffic_list[1]
                r = traffic_list[2]
                if max(l, max(n, r)) == traffic_list[0]:
                    left += 1
                elif max(l, max(n, r)) == traffic_list[1]:
                    none += 1
                elif max(l, max(n, r)) == traffic_list[2]:
                    right += 1
    if max(left, right) == left:
        return np.array([0, 1, 0])
    elif max(left, right) == right:
        return np.array([0, 0, 1])
    elif left and right == none:
        return np.array([0, 0, 0])


def show(img):
    cv2.imshow('acac', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generator(type_data, path, batch_size, img_shape):
    img_rgb_path, img_depth_path, angles = load_data(type_data=type_data, path=path)
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
        traffics_list = []
        for index in range(batch_size):
            count = 0
            file_name = x[index].split('/')[-1]
            for each in file_name:
                if each == '_':
                    count += 1
            img_rgb = cv2.imread(x[index])

            if count == 2:
                img_rgb = augment_object.augment_image(img_rgb)
            else:
                img_rgb = augment_min.augment_image(img_rgb)
             # predict
            traffic_status = get_predict(img_rgb)
            traffics_list.append(traffic_status)
            h, w, _ = img_rgb.shape
            img_rgb = img_rgb[h//2:h, :w]
            img_rgb = cv2.resize(img_rgb, img_shape[:-1])
            # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            # img_depth = cv2.imread(x2[index], 0)
            # hd, wd = img_depth.shape
            # img_depth = img_depth[hd//2:hd, :wd]
            # img_list.append(np.dstack((img_rgb, img_depth)))
            img_list.append(img_rgb)
            angle = y1[index]
            angle_list.append(angle)

        traffics_list = np.array(traffics_list)
        img_list = np.array(img_list)
        angle_list = np.array(angle_list)
        # print(angle_list.shape)
        # print(traffics_list.shape)
        # print(img_list.shape)
        # inputs = []
        # inputs.append(np.dstack((img_list,traffics_list)))
        # inputs = np.array(inputs)
        yield [img_list, traffics_list], angle_list


def get_callback(weight_path, batch_size):
    # earlystop = EarlyStopping(
    #     monitor='val_loss', patience=5, verbose=0, mode='min')
    tensorboard = TensorBoard(log_dir="logs/resnet50_{}".format(time.time()),
                              batch_size=batch_size, write_images=True)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    callbacks = [tensorboard, checkpoint, reduce_lr]
    return callbacks
