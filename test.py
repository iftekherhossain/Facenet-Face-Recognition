from tensorflow.keras.models import load_model
import os
import cv2
from face_main import Face_utils
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import *
f = Face_utils()
base_path = "D:\\Facenet-Face_recognition\\faces\\val\\"
haar_path = "D:\\Facenet-Face_recognition\\haarcascade_frontalface_default.xml"
faces = os.listdir(base_path)
dataz = np.load('mean_embeddings.npz')
mean_emds = dataz['arr_0']
labels = dataz['arr_1']
final_labels = [faces[q] for q in labels]
print(final_labels)


model = load_model("facenet_keras.h5")

w=1
comp_emd = mean_emds[w]
comp_label = final_labels[w]
X_ = []
y_ = []
for face in faces:
    temp_path = base_path+"\\"+face
    li = os.listdir(temp_path)
    for l in li:
        try:
            full_path = temp_path+"\\"+l
            img = cv2.imread(full_path)
            boxes = f.detect_face_haar_cascade(haar_cascade_path=haar_path,image=img)
            box = boxes[0]
            roi = f.return_face(img, box)
            temp_emd = f.face_embedding(model, roi)
            y = f.compare_embeddings(comp_emd,temp_emd)
            y_.append(y)
            X_.append(l[:-4])
            # cv2.imshow("kuki",face)
            # cv2.waitKey(0)
        except:
            pass

print(X_)
print(y_)
fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.bar(X_, y_,color = 'b')
# plt.show()
X_pos = range(len(X_))
plt.bar(X_,y_)
plt.xticks(X_pos, X_, rotation=90)
plt.show()