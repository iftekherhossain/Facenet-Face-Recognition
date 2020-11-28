import dlib
import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import FPS
import os
from imutils.video import FileVideoStream
import time
from scipy.spatial import distance
#-----------------------------------------------------#
# load mean_embeddings
emds_and_labels = np.load("data.npz")
embeddings = emds_and_labels["arr_0"]
labels = emds_and_labels["arr_1"]
print("labels", labels)
#-----------------------------------------------------#
# cap = cv2.VideoCapture('test\\five.mp4')
fvs = FileVideoStream("test\\five.mp4").start()
time.sleep(1.0)
#-----------------------------------------------------#
f = Face_utils()
#-----------------------------------------------------#
model = load_model("facenet_keras.h5")

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
image = cv2.imread("test/bean.jpg")
boxes = f.detect_face_dnn(net, image, con=0.5)
box = boxes[0]
x, y, w, h = box[0], box[1], box[2], box[3]
tup_box = (x, y, w, h)
cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
face = f.return_face(image, tup_box)
real_emd = f.face_embedding(model, face)
c = 0
#------------------------------------------------------------#
#--------------------for framerate---------------------------#
fps = FPS().start()
#------------------------------------------------------------#
start = time.time()
total_comapare_time = 0
while True:
    fps.update()
    c += 1
    # ret, frame = cap.read()
    frame = fvs.read()
    try:
        boxes = f.detect_face_dnn(net, frame, con=0.9)
    except:
        break
    check_tuple = type(boxes) is tuple
    # print(boxes)
    if len(boxes) == 0 and not check_tuple:
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    elif len(boxes) >= 1 and not check_tuple:
        for box in boxes:
            #box = boxes[0]
            x, y, w, h = box[0], box[1], box[2], box[3]
            tup_box = (x, y, w, h)
            # print(tup_box)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            try:
                face = f.return_face(frame, tup_box)
            except:
                continue
            real_emd = f.face_embedding(model, face)
            preds = []
            # print(len(embeddings))
            s = time.time()
            for fin_emd in embeddings:
                q = f.compare_embeddings(real_emd, fin_emd)
                # q = distance.euclidean(real_emd, fin_emd)
                preds.append(q)
            e = time.time()
            tc = e-s
            total_comapare_time += tc
            #print("SSSSSSSSSSSS", e-s)
            lowest_index = np.argmin(preds)
            # print(preds)
            if preds[lowest_index] < threshold:
                b = int_to_name[lowest_index]
                print(int_to_name[lowest_index])
                f.draw_text(frame, b, (x+10, y-10))
            else:
                print("Unknown")
                f.draw_text(frame, "Unknown", (x+10, y+10))
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
end = time.time()
print("Final_time", end-start)
print("Total Compare Time", total_comapare_time)
print("Total Frames", c)
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# When everything is done, release the capture
# cap.release()

cv2.destroyAllWindows()
