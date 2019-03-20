import cv2
import numpy as np
# import glob
import json

data_path = '/home/linus/Desktop/data3/combined.json'
# glob.glob(data_path + '*.jpg')
dataset = json.loads(open(data_path, 'r').read())


def circle_to_bb(img, x, y, r, index, index_phu):
    top_y = max(y - r - 10, 0)
    top_x = max(x - r - 10, 0)
    y_size = min(top_y+r*2+20, img.shape[0])
    x_size = min(top_x+r*2+20, img.shape[1])
    print(top_x, top_y)
    print(img.shape)
    img = img[top_y:y_size, top_x:x_size, :]
    print(img.shape)
    cv2.imwrite('./traffic_sign/rgb2_{}_{}.jpg'.format(index, index_phu), img)
    cv2.imshow('bb', img)



for index, sample in enumerate(dataset):
    each_file_path = sample['rgb_img_path'].replace(
        '/Desktop/data/', '/Desktop/data3/')
    print(each_file_path)
    img = cv2.imread(each_file_path)
    s = img.shape
    img = img[:s[0]//2,:]
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

    # ensure at least some circles were found
    if circles is not None and np.sum(circles) > 0:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        print('Got', len(circles), 'circles')

        # loop over the (x, y) coordinates and radius of the circles
        for index_phu, (x, y, r) in enumerate(circles):
            print(x, y, r)
            # print(x, y, r)
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 255), 4)
            circle_to_bb(raw, x, y, r, index, index_phu)
        # show the output image
    else:
        print('No circle')
    cv2.imshow("color", color)
    cv2.imshow("output", output)
    cv2.imshow('lol', res)
    cv2.waitKey(1)


cv2.destroyAllWindows()
