"""
This python script is made for the horrible names of the files of  5 celebraty dataset
"""
import numpy as np
import os 
import cv2
from face_main import Face_utils

f= Face_utils()

train_path = "D:\\Facenet-Face_recognition\\faces\\train"
test_path = "D:\\Facenet-Face_recognition\\faces\\val"
train_li = os.listdir(train_path)
test_li = os.listdir(test_path)
for sub_dir_train in train_li:
    a_train = sub_dir_train
    full_path_train = train_path + "\\" + sub_dir_train
    face_li_train = os.listdir(full_path_train)
    i=1
    for im_train in face_li_train:
        os.rename(full_path_train+"\\"+im_train,full_path_train+"\\"+a_train+str(i)+".jpg")
        i+=1

for sub_dir_test in test_li:
    a_test = sub_dir_test
    full_path_test = test_path + "\\" + sub_dir_test
    face_li_test = os.listdir(full_path_test)
    i=1
    for im_test in face_li_test:
        os.rename(full_path_test+"\\"+im_test,full_path_test+"\\"+a_test+str(i)+".jpg")
        print("s")
        i+=1
