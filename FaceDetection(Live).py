'''
Here in this code where i have tried to detect the face of a person on live video using the ai model (Haarcascade).
'''

#Imported opencv as cv
import cv2 as cv

#initializing the camera module
capture = cv.VideoCapture(0)

#Initializing the Haarcascade model
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'Haarcascade_frontalface_default.xml')

#Running the infinite loop to load all the frames using while loop
while True:
    capture_status, frame = capture.read()

    #If the frame are not captured properly then the loop is terminated
    if not capture_status: 
        break

    #Converting the every frame from BGR to Gray
    Gray_Frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Detecting the face using the haarcascade model
    face_roi = face_cascade.detectMultiScale(Gray_Frame, scaleFactor=1.3, minNeighbors=5)

    #Writing the text for proper labeling
    text = "Face"

    #Running loops in the face_roi to extract the required co-ordinates of face
    for (x, y, w, h) in face_roi:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv.putText(frame, text, (x, y-10), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 2)
    
    #Displaying the live video with detected face
    cv.imshow("Face Detected Image", frame)

    #Exting the loop
    if cv.waitKey(1) & 0xFF == ord('k'):
        break

#Make sure the camera module is trunned off after the loop is terminated
capture.release()

#Ensuring nothing is running in background after the process is completed
cv.destroyAllWindows()