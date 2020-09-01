import dlib
import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import FPS
#----#
from sklearn.svm import SVC
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#---#
import os 

#-----------------------------------------------------#
#load mean_embeddings
# emds_and_labels = np.load("data.npz")
# embeddings = emds_and_labels["arr_0"]
#labels = os.listdir()
#print("labels",labels)
#-----------------------------------------------------#
cap = cv2.VideoCapture(1)
#-----------------------------------------------------#
f= Face_utils()
#-----------------------------------------------------#
model = load_model("facenet_keras.h5")
cascade_path = "haarcascade_frontalface_default.xml"
#-----------------------------------------------------#
#create dictionary for int to label 
#int_to_name = {0:"ben_afflek",1:"elton_john",2:"jerry_seinfeld",3:"madonna",4:"mindy_kaling",5:"Nabil"}
threshold = 12
faces =os.listdir("faces//train")
int_to_name = {}
for i,face in enumerate(faces):
    int_to_name[i]=face
print(int_to_name)
print(faces)
#--------------------For SVM---------------------------------#
data = np.load("data.npz")
y_train = data['arr_1']
y_test = data['arr_3']
X_train = data['arr_0']
X_test =  data['arr_2']
label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(y_train)
clf = make_pipeline(sk.svm.LinearSVC(C=0.1))
clf.fit(X_train,y_train_int)
#-----------For dnn face detection---------------------------#
path_proto = 'D:\\Facenet-Face_recognition\\deploy.prototxt.txt'
path_model = 'D:\\Facenet-Face_recognition\\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#------------------------------------------------------------#
#--------------------for framerate---------------------------#
fps = FPS().start()
#------------------------------------------------------------#
while True:
    fps.update()
    ret, frame = cap.read()
    boxes = f.detect_face_dnn(net,frame,con=0.9)
    check_tuple = type(boxes) is tuple
    #print(boxes)
    if len(boxes)>=1 and not check_tuple:
        box = boxes[0]
        x,y,w,h = box[0],box[1],box[2],box[3]
        tup_box = (x,y,w,h)
        #print(tup_box)
        if w>120 and h >120:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face = f.return_face(frame,tup_box)
            real_emd = f.face_embedding(model,face)
            real_emd = np.reshape(real_emd,(1,128))
            preds = clf.predict(real_emd)
            pred = preds[0]
            b = int_to_name[pred]
            f.draw_text(frame,b,(x+10,y-10))
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows() 