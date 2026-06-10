#Importing the open cv module
import cv2 as cv

#Importing the numpy module
import numpy as np

#Importing the OS module
import os

#Loading the Haarcascade xml file
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        #Implementing the haarcascade
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            #Cropping and resizing image
            face_crop = image[y:y+h, x:x+w]
            resized_face = cv.resize(face_crop, (200, 200))

            #appending the resized face into the faces list
            Faces.append(resized_face)
            Labels.append(current_label)

    #To store the next image_name
    current_label += 1

#Representing the list as array using np
Final_Faces = np.array(Faces)
Final_Labels = np.array(Labels)

#Tranning the recognizer
recognizer.train(Final_Faces, Final_Labels)
print('Successfully Trainned the images.')

#Reading the test images for LBPH
Test_Image = cv.imread(r'Tranning_Images/Xxxx_Ten/image copy 8.png')

#Converting the test image to gray
T_Img_Gray = cv.cvtColor(Test_Image, cv.COLOR_BGR2GRAY)

#Tracking Face in the test image
T_Img_Face = face_cascade.detectMultiScale(T_Img_Gray, scaleFactor=1.2, minNeighbors=5)

#Cropping and resizing the face from the Test_Image
for (x,y,w,h) in T_Img_Face:
    cropped_T_Img = T_Img_Gray[y:y+h, x:x+w]
    resized_T_Img = cv.resize(cropped_T_Img, (200, 200))

    label, confidence = recognizer.predict(resized_T_Img)
    person_name = label_to_face[label]

    #Writing the text for proper labeling
    Text = f'{person_name} ({round(confidence)})'

    #Drawing the rectangle around the face of test image
    cv.rectangle(Test_Image, (x, y), (x+w, y+h), (0,255,255), 2)

    #Including the proper label for efficient recognization
    cv.putText(Test_Image, Text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

#Displaying the test image and run the LBPH
cv.imshow("Test Image", Test_Image)
cv.waitKey(0)
cv.destroyAllWindows()
