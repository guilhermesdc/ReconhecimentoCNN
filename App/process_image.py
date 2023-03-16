import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

classes = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'surprised': 3,
    'fearful': 4,
    'disgusted': 5,
    'angry': 6,
    'contempt': 7
}

def pre_process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y + h, x:x + w]
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
        img = cv2.equalizeHist(img)
        img = img.astype('float32') / 255
        img = np.expand_dims(img, -1)
        return img
    else:
        return None

def get_emotion(img):
    img = img.read()
    img = np.fromstring(img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = pre_process_image(img)
    if img is None:
        return 'No face detected'
    img = np.expand_dims(img, 0)
    model = tf.keras.models.load_model('cnn_model.h5')
    result = model.predict(img)
    result = np.argmax(result)
    result = list(classes.keys())[list(classes.values()).index(result)]
    return result