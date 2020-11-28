from tensorflow.keras.models import load_model
from face_main import Face_utils
import cv2
import numpy as np
import pickle
import os

emds_and_labels = np.load("mean_embeddings.npz")
embeddings = emds_and_labels["arr_0"]
labels = emds_and_labels["arr_1"]
print("labels", labels)

faces = os.listdir("faces//val")
int_to_name = {}
for i, face in enumerate(faces):
    int_to_name[i] = face
int_to_name[6] = "Unknown"
model = load_model('facenet_keras.h5')

path = "D:\\Facenet-Face_recognition\\test\\nabil1.jpg"

f = Face_utils()
#-----------For dnn face detection---------------------------#
path_proto = 'D:\\Facenet-Face_recognition\\deploy.prototxt.txt'
path_model = 'D:\\Facenet-Face_recognition\\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#------------------------------------------------------------#

image = cv2.imread(path)
boxes = f.detect_face_dnn(net, image)

box = boxes[0]
x, y, w, h = box[0], box[1], box[2], box[3]

roi = f.return_face(image, box)
embedding = f.face_embedding(model, roi)
for i, emd in enumerate(embeddings):
    a = f.compare_embeddings(emd, embedding)
    if a < 12:
        lowest_index = i
    else:
        lowest_index = 6
        continue
b = int_to_name[lowest_index]
print(b)
cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
# f.draw_text(image, b, (x+40, y-10))
f.draw_text(image, "SSD", (10, 30))
cv2.imshow("SSD", image)
cv2.waitKey(0)
# print(face_pix.shape)
#print("sample ", sample.shape)
print(embedding)
