from kafka import KafkaConsumer
import cv2
import numpy as np
import datetime

consumer = KafkaConsumer('my-topic', bootstrap_servers='localhost:9092')
faceCascade= cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


for message in consumer:
    array = np.frombuffer(message.value, dtype = np.dtype('uint8'))
    img = cv2.imdecode(array, 1)
 
    # Code for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
         gray,
         scaleFactor=1.3,
         minNeighbors=5,
         minSize=(30, 30)
         )
    for(x, y, w, h) in faces:
         cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

    print("Face Count : {0}".format(len(faces)))
    # Streaming
    cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Image', img)
    
    # Quit streaming
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
