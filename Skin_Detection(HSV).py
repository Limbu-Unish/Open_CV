'''
Here in this code i have wriiten a code which takes the raw image as an input and using (HSV, Masking) in output only the skin area are noticable/visible
'''

#importing the opencv as cv
import cv2 as cv

#Importing the numpy as np
import numpy as np

#Initializing the image path
image_path = 'Test_Images/image.png'

#importing the testing image
image = cv.imread(image_path)

#Converting the BGR to HSV formate
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#Initializing the lower and upper skin color range
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype= np.uint8)

#initializing the masking
mask = cv.inRange(hsv_image, lower_skin, upper_skin)

#Detecting skin using bitwise_and
detected_skin = cv.bitwise_and(image, image, mask=mask)

#Displaying the original image
cv.imshow("Original_Image", image)

#Displaying the mask image
cv.imshow('Masked Image', mask)

#Displaying the original image
cv.imshow('Detected Skin', detected_skin)

cv.waitKey(0)
cv.destroyAllWindows()