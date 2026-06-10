'''
Here in this code, i have trained the images so, that the face can be deteced in the live videos
'''

#Importing cv2 (Opencv) as cv
import cv2 as cv

#Importing the numy as np
import numpy as np

#Importing the os module
import os

#Implementing the LBPH algorithm
recognizer = cv.face.LBPHFaceRecognizer_create()

#Implementing the Haarcascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'Haarcascade_frontalface_default.xml')

#creating separeate empty list for the faces and labels
faces = []
labels = []

#Creating a dictionary for paring the face and names
label_to_name ={}

#Initial Label
current_label = 0

#Trained image path
Trained_img_path = 'Tranning_Images'

#Iterating over every images inside the tranning_image folder for cropping anf resizing
for person_name in os.listdir(Trained_img_path):
    person_folder_path = os.path.join(Trained_img_path, person_name)

    label_to_name[current_label] = person_name

    for person_img in os.listdir(person_folder_path):
        person_img_path = os.path.join(person_folder_path, person_img)

        
        image = cv.imread(person_img_path, cv.IMREAD_GRAYSCALE)
        face = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in face:
            cropped_face = image[y:y+h, x:x+w]
            resized_face = cv.resize(cropped_face, (200, 200))

            faces.append(resized_face)
            labels.append(current_label)
    
    current_label += 1

Final_Faces = np.array(faces)
Final_Labels = np.array(labels)


recognizer.train(Final_Faces, Final_Labels)
print('Images trained successfully.')

#Capturing live videos
live_capture = cv.VideoCapture(0)

while True:
    capture_status, frame = live_capture.read()

    if not capture_status: break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    L_face = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in L_face:
        cropped_face = gray_frame[y:y+h, x:x+w]
        resized_face = cv.resize(cropped_face, (200, 200))

        label, confidence =recognizer.predict(resized_face)

        if label in label_to_name:
            person_name = label_to_name[label]
        else:
            person_name = 'Unknown'

        text = f'{person_name}: {round(confidence, 2)}'

        cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        cv.putText(frame, text, (x, y-10), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,255), 2)
        cv.imshow("Live Video", frame)
    if cv.waitKey(1) & 0xFF == ord('k'):
        break

#Closing the camera module
live_capture.release()

#Destroying the unnecessary operation running in the background
cv.destroyAllWindows()
