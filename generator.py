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

# Python generator to generate data for training on the fly, rather than storing everything in memory
def generator(samples, sample_size, mode):
    # Total number of samples
    num_samples = len(samples)
    # Generator loop
    while True:
        # Shuffle data set for every batch
        sklearn.utils.shuffle(samples, random_state=43)
        
        # Go through one batch with batch_size steps
        for offset in range(0, num_samples, sample_size):
            # Get the samples for one batch
            batch_samples = samples[offset:offset+sample_size]
            
            # Store the images and angles and speed 
            images = []
            angles = []
            speed = []
            # Get the images for one batch
            for sample in batch_samples:
                # Get the different camera angles for every sample
            
                    # Choose randomly what kind of augmentation to use
                if mode == 'train':
                    augmentation = np.random.choice(['flipping', 'brightness', 'shift', 'none'])
                else:
                    augmentation = 'none'
                    
                # Image and angle
                image = None
                if sample[3] =='steering':
                    print('okkk')
                angle = float(sample[3])
                    #Load speed 
                speed = float(sample[6])
               
                image = cv2.imread('./data/' + sample[0])

                    # # Load the left image and alter angle
                    # elif cam == 'left':
                    #     image = cv2.imread('./data/' + sample[1].strip())
                    #     angle += 0.2
                                             
                    # # Load the right image and alter angle
                    # elif cam == 'right':
                    #     image = cv2.imread('./data/' + sample[2].strip())
                    #     angle -= 0.2
                    
                    #cv2.imshow('ace',image)
                    #cv2.waitKey(1000)
                    # Convert the image to RGB
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                #print(image.shape())
                # Flip the image and correct angle
                if augmentation == 'flipping':
                    image = cv2.flip(image, 1)
                    angle *= -1.0
                # Change the brightness of the image randomly
                elif augmentation == 'brightness':
                    # Convert to HSV color space
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                    # Extend range
                    image = np.array(image, dtype = np.float64)
                    # Choose new value randomly
                    brightness = np.random.uniform() + 0.5
                    # Alter value channel
                    image[:,:,2] = image[:,:,2] * brightness
                    # When value is above 255, set it to 255
                    image[:,:,2][image[:,:,2] > 255] = 255
                    # Convert back to 8-bit value
                    image = np.array(image, dtype = np.uint8)
                    # Convert back to RGB color space
                    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                        
                        # Apply a random shift to the image
                elif augmentation == 'shift':
                    # Translation in x direction
                    trans_x = np.random.randint(0,100) - 50
                     # Correct angle
                    angle += trans_x * 0.004
                        # Translation in y direction
                    trans_y = np.random.randint(0,40)- 20
                    # Create the translation matrix
                    trans_matrix = np.float32([[1,0,trans_x],[0,1,trans_y]])
                    image = cv2.warpAffine(image,trans_matrix,(320, 160))
                        
                # Crop the image
                image = image[50:140, 0:320]
                    
                # Resize to 200x66 pixel
                image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
                        
                # Add image and angle to the list
                images.append(np.reshape(image, (1, 66,200,3)))
                angles.append(np.array([[angle]]))
                
            #print(angles)        
            # Return the next batch of samples shuffled
            X_train = np.vstack(images)
            y_train = np.vstack(angles)
            #print(y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train, random_state=21)