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

#Created a list for storing the faces
Faces = []

#Created a list for storing the labels relevant to the faces
Labels = []

#Created a dictionary to store key value pair for faces and labels
label_to_face = {}

current_label = 0

#Looping inside the tranning_images folder
for person_name in os.listdir(data_set_path):
    person_folder_path = os.path.join(data_set_path, person_name)

    label_to_face[current_label] = person_name

    #looping inside the particular preson image folder
    for person_img in os.listdir(person_folder_path):
        img_path = os.path.join(person_folder_path, person_img)

        #Reading the image
        image = cv.imread(r'img_path', cv.IMREAD_GRAYSCALE)

        #Implementing the haarcascade
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            #Cropping and resizing image
            face_crop = image[y:y+h, x:x+w]
            resized_face = cv.resize(face_crop, (300, 300))

            #appending the resized face into the faces list
            Faces.append(resized_face)
            Labels.append(current_label)

    current_label += 1     