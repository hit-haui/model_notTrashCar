import cv2
import numpy as np
# import glob


data_path = '/Users/lamhoangtung/Downloads/dataset_1552725149.9200957/rgb/'
# glob.glob(data_path + '*.jpg')

for index in range(1, 292):
    each_file_path = data_path + '{}_rgb.jpg'.format(index)
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
        print(circles)
        circles = np.round(circles[0, :]).astype("int")
        print('Got', len(circles), 'circles')

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # print(x, y, r)
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 255), 4)
        # show the output image
    else:
        print('No circle')
    cv2.imshow("color", color)
    cv2.imshow("output", output)
    cv2.imshow('lol', res)
    cv2.waitKey(100)


cv2.destroyAllWindows()
