import cv2
import numpy as np 
import os
path = '/home/vicker/Downloads/dataset_1552725149.9200957/'
for filename in os.listdir(path+'rgb/'):
    print(filename)
    img = cv2.imread(path+'rgb/'+filename)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #100,35,35
    lower_blue = np.array([100,80,50])
    upper_blue = np.array([120,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow('img',img)
    # cv2.imshow('mask',mask)
    h,w,_ = img.shape
    cv2.imwrite(path+'detected/'+filename,res[80:int(h/2),:w])
    
