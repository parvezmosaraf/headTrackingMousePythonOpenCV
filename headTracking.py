import cv2
import numpy as np
from pynput.mouse import Button, Controller

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouse = Controller()

cap = cv2.VideoCapture(0)

# Define the clicking function
def perform_click(x, y):
    mouse.position = (x, y)
    mouse.click(Button.left, 1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Find the center of the face
        center = (x + w//2, y + h//2)

        # Move the mouse cursor to the center of the face
        mouse.position = center

        # Perform a left click at the center of the face
        perform_click(*center)

    cv2.imshow("Head Tracking", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
