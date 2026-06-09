#Importing the open cv module
import cv2 as cv

#Importing the numpy module
import numpy as np

#Importing the OS module
import os

#Loading the Haarcascade xml file
face_cascade = cv.CascadeClassifier(cv.data.haarcasades + 'haarcascade_frontalface_default.xml')

#Loading the LBPH recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()

#Initializing the data set folder path
data_set_path = 'Tranning_Images'

Faces = []
Labels = []