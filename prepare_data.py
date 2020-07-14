import numpy as np
import os 
import cv2
from face_main import Face_utils
from tensorflow.keras.models import load_model

f= Face_utils()
train_path = "faces\\train"
test_path = "faces\\val"
li_train = os.listdir(train_path)
li_test = os.listdir(test_path)
X_train = []
y_train = []
X_test = []
y_test = []
model = load_model('facenet_keras.h5')
i,j=0,0
for sub_dir_train in li_train:
    a_train = sub_dir_train
    full_path_train = train_path + "\\" + sub_dir_train
    face_li_train = os.listdir(full_path_train)
    for im_train in face_li_train:
        k = full_path_train+ "\\"+ im_train
        image= cv2.imread(k)
        boxes = f.detect_face(k)
        if len(boxes)>=1:
            box = boxes[0]
            face = f.return_face(image,box)
            emd = f.face_embedding(model, face)
            X_train.append(emd)
            y_train.append(a_train)
            print("processing traing..."+str(i))
            i+=1
        else:
            continue

for sub_dir_test in li_test:
    a_test = sub_dir_test
    full_path_test = test_path + "\\" + sub_dir_test
    face_li_test = os.listdir(full_path_test)
    for im_test in face_li_test:
        k = full_path_test+ "\\"+ im_test
        image= cv2.imread(k)
        boxes = f.detect_face(k)
        if len(boxes)>=1:
            box = boxes[0]
            face = f.return_face(image,box)
            emd = f.face_embedding(model, face)
            X_test.append(emd)
            y_test.append(a_test)
            print("processing test..."+str(j))
            j+=1
        else:
            continue

np.savez_compressed('data.npz',X_train,y_train,X_test,y_test)