from kafka import KafkaConsumer
from flask import Flask, Response
import cv2
import numpy as np
import datetime

consumer = KafkaConsumer('my-topic', bootstrap_servers='localhost:9092')
faceCascade= cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
eyeCascade= cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

app = Flask(__name__)


def kafkastream():
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
        # Draw rectangle around detected faces
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            # For eye detection
            eyes = eyeCascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
            # For each eyes, draw rectangle
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex,ey), (ex+ew, ey+eh),(0,255,0), 2)

        msg = cv2.imencode('.jpeg', img)[1].tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + msg + b'\r\n\r\n')

@app.route('/')
def index():
    return Response(kafkastream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
