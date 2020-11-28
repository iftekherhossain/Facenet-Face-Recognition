import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import FPS
import os
import time

#-----------------------------------------------------#
# load mean_embeddings
final_emd_and_labels = np.load("data.npz")
final_embeddings = final_emd_and_labels["arr_0"]
labels = final_emd_and_labels["arr_1"]
print("labels", labels)
#-----------------------------------------------------#
cap = cv2.VideoCapture("mindy.mp4")
#-----------------------------------------------------#
f = Face_utils()
#-----------------------------------------------------#
model = load_model("facenet_keras.h5")
cascade_path = "haarcascade_frontalface_default.xml"
#-----------------------------------------------------#
# create dictionary for int to label
#int_to_name = {0:"ben_afflek",1:"elton_john",2:"jerry_seinfeld",3:"madonna",4:"mindy_kaling",5:"Nabil"}
threshold = 12
faces = labels
int_to_name = {}
for i, face in enumerate(faces):
    int_to_name[i] = face
print(int_to_name)
print(faces)
#-----------For dnn face detection---------------------------#
path_proto = 'D:\\Facenet-Face_recognition\\deploy.prototxt.txt'
path_model = 'D:\\Facenet-Face_recognition\\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#------------------------------------------------------------#
#--------------------dummy recognition-----------------------#
image = cv2.imread("test/bean.jpg")
boxes = f.detect_face_dnn(net, image, con=0.5)
box = boxes[0]
x, y, w, h = box[0], box[1], box[2], box[3]
tup_box = (x, y, w, h)
cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
face = f.return_face(image, tup_box)
real_emd = f.face_embedding(model, face)
#--------------------for framerate---------------------------#
performance_test = []
data_path = 'D:\\Facenet-Face_recognition\\test\\ben'
pics = os.listdir(data_path)
print(pics)
target_class = 'elton_john'
i = 0
unknown = []
different_class = []
no_detect_faces = []
for pic in pics:
    print(pic)
    full = data_path+"//"+pic
    image = cv2.imread(full)
    image = cv2.resize(image, (300, 300))
    boxes = f.detect_face_dnn(net, image)
    check_tuple = type(boxes) is tuple
    # print(boxes)
    if len(boxes) >= 1 and not check_tuple:
        box = boxes[0]
        x, y, w, h = box[0], box[1], box[2], box[3]
        tup_box = (x, y, w, h)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = f.return_face(image, tup_box)
        real_emd = f.face_embedding(model, face)
        preds = []
        for fin_emd in final_embeddings:
            q = f.compare_embeddings(real_emd, fin_emd)
            preds.append(q)
        lowest_index = np.argmin(preds)
        print("Preds", preds, len(preds))
        # print(lowest_index)
        if preds[lowest_index] < threshold:
            b = int_to_name[lowest_index]
            print(int_to_name[lowest_index])
            f.draw_text(image, b, (x+10, y-10))
            if b != target_class:
                tt = (pic, b)
                different_class.append(tt)
        else:
            print("Unknown")
            f.draw_text(image, "Unknown", (x+10, y+10))
            unknown.append(pic)
        cv2.waitKey(1)
        cv2.imshow("hello", image)
        cv2.imwrite("D:\\Facenet-Face_recognition\\test\\hudai\\"+pic, image)
        # time.sleep(1)
    elif len(boxes) == 0:
        no_detect_faces.append(pic)
print("different_class", different_class, len(different_class))
print("Unknown", unknown, len(unknown))
print("No detect face", no_detect_faces, len(no_detect_faces))
