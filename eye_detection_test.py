import numpy as np
import cv2

# Import face recognizer
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

eyeCascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
# Read saved image
image = cv2.imread('test2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Face recognition
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Print number of detected faces
print("Face count: {0}".format(len(faces)))

# Draw rectangle around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
# Show result
while True:
    cv2.imshow("Face", image)
    # Press 'q' to close window
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
