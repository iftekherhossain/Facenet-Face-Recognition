import dlib
import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN

image_path = "nabil1.jpg"
haar_cascade_path= "haarcascade_frontalface_default.xml"
path_proto = 'D:\\Facenet-Face_recognition\\deploy.prototxt.txt'
path_model = 'D:\\Facenet-Face_recognition\\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
image = cv2.imread(image_path)

f = Face_utils()
detector = MTCNN()
name = "MTCNN"

boxes = f.detect_face_mtcnn(detector,image)
box = boxes[0]
x,y,w,h = box[0],box[1],box[2],box[3]
tup_box = (x,y,w,h)
cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
f.draw_text(image,name,(10,30))
cv2.imshow(name,image)
cv2.waitKey(0)