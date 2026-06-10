'''
Here in this code where i have tried to detect the face of a person on static image using the ai model (Haarcascade).
'''

#Imported opencv as cv
import cv2 as cv

#Initializing the test imsge to detect the face inside it
image = cv.imread(r'Tranning_Images/Rajesh_Hamal/image copy.png')

#Converting the image from BGR to Gray
image_Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#Initializing the Haarcascade model
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'Haarcascade_frontalface_default.xml')

#Detecting the face using the haarcascade model
face_roi = face_cascade.detectMultiScale(image_Gray, scaleFactor=1.3, minNeighbors=5)

#Writing the text for proper labeling
text = "Face"

#Running loops in the face_roi to extract the required co-ordinates of face
for (x, y, w, h) in face_roi:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv.putText(image, text, (x, y-10), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 255), 2)

#Displaying the image with detected face
cv.imshow("Face Detected Image", image)
cv.waitKey(0)
cv.destroyAllWindows()