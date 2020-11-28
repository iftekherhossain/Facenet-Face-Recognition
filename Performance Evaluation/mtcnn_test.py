from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import FPS
import os 

img = cv2.imread('nabil1.jpg')


detector = MTCNN()
# faces = detector.detect_faces(img)
# for face in faces:
#     print(face)

def detect_face_mtcnn(detector,image):
    faces = detector.detect_faces(image)
    boxes = []
    for face in faces:
        boxes.append(face['box'])
    return boxes
boxes = detect_face_mtcnn(detector,img)