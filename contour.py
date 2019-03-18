import cv2
import os 
import numpy as np 

path = '/home/vicker/Downloads/dataset_1552725149.9200957/'
for filename in os.listdir(path+'detected/'):
    print(filename)
    img = cv2.imread(path+'detected/'+filename)
    #img = cv2.imread('/home/vicker/Downloads/cC8if.jpg')
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_predict = cv2.imread(path+'rgb/'+filename)

    (ret,thresh) = cv2.threshold(imgray,90,130,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    for i in range(len(contours)):
        cnt = contours[i]
        ##bb
        # x,y,w,h = cv2.boundingRect(cnt)
        # if w>9 and w < 40 and h > 9 and h < 40:
        #     cv2.rectangle(img_predict,(x,y+80),(x+w,y+80+h),(0,255,0),2)

        ##circle
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        if radius > 11 and radius <30:
            cv2.circle(img,center,radius,(0,255,0),2)
            
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(img,[box],0,(0,0,255),2)

    #cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imwrite(path+'contour/'+filename,img)





# #cimg = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# cimg = mask
# circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,10,
#                             param1=50,param2=30,minRadius=0,maxRadius=10)

# circles = np.uint16(np.around(circles))
# print(img.shape)
# print(circles)
# for i in circles[0,:]:
#     # draw the outer circle
#     #cv2.imshow('circle',img[])
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
 
# cv2.imshow('anc',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()