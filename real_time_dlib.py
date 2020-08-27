import dlib
import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import FPS
import os 

#-----------------------------------------------------#
#load mean_embeddings
final_emd_and_labels = np.load("mean_embeddings.npz")
final_embeddings = final_emd_and_labels["arr_0"]
labels = final_emd_and_labels["arr_1"]
print("labels",labels)
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
faces = os.listdir("faces\\train\\")
int_to_name = {}
for i,face in enumerate(faces):
    int_to_name[i]=face
print(int_to_name)
print(faces)
#--------------------for framerate---------------------------#
fps = FPS().start()
#------------------------------------------------------------#
while True:
    fps.update()
    ret, frame = cap.read()
    boxes = f.detect_face_dlib(frame)
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
            preds =[]
            for fin_emd in final_embeddings:
                q = f.compare_embeddings(real_emd,fin_emd)
                preds.append(q)
            lowest_index = np.argmin(preds)
            print(preds)
            if preds[lowest_index]<threshold:
                b = int_to_name[lowest_index]
                print(int_to_name[lowest_index])
                f.draw_text(frame,b,(x+10,y-10))
            else:
                print("Unknown")
                f.draw_text(frame,"Unknown",(x+10,y+10))
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