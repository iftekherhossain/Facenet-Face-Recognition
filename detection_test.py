import dlib
import cv2
from face_main import Face_utils
import numpy as np
from imutils.video import FPS
from mtcnn.mtcnn import MTCNN
import os 

cap = cv2.VideoCapture(1)
f = Face_utils()
#-----------For dnn face detection---------------------------#
path_proto = 'D:\\Facenet-Face_recognition\\deploy.prototxt.txt'
path_model = 'D:\\Facenet-Face_recognition\\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#------------------------------------------------------------#
#----------------For Haar_cascade detection------------------#
cascade_path = "haarcascade_frontalface_default.xml"
#------------------------------------------------------------#
#--------------------detector_mtcnn--------------------------#
detector = MTCNN()
#-----------------------------------------------------------#
fps = FPS().start()
while True:
    fps.update()
    ret, frame = cap.read()
    boxes = f.detect_face_mtcnn(detector,frame)
    try:
        box = boxes[0]
        x,y,w,h = box[0],box[1],box[2],box[3]
        tup_box = (x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))