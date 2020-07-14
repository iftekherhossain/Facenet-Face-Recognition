import cv2
import dlib
import numpy as np
from imutils import face_utils

class Face_utils:
    @staticmethod
    def detect_face(face_path):
        detector = dlib.get_frontal_face_detector()
        img = dlib.load_rgb_image(face_path)
        dets = detector(img,1)
        boxes=[]
        for i, d in enumerate(dets):
            (x,y,w,h)= face_utils.rect_to_bb(d)
            box = (x,y,w,h)
            boxes.append(box)
        return boxes

    @staticmethod
    def detect_face_haar_cascade(haar_cascade_path,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(haar_cascade_path)

        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if (len(faces)==0):
            return None, None
        return faces


    @staticmethod
    def return_face(image,box):
        x, y, w, h = box
        roi = image[y:y+h,x:x+w]
        roi_resize = cv2.resize(roi,(160,160))
        return roi_resize

    @staticmethod
    def face_embedding(model,roi):
        roi = np.array(roi)
        face_pix = roi.astype('float32')
        mean, std = face_pix.mean(), face_pix.std()
        face_pix = (face_pix-mean) / std
        sample = np.expand_dims(face_pix,axis=0)
        emd = model.predict(sample)
        return emd[0]
    
    @staticmethod
    def compare_embeddings(emd1,emd2):
        return np.linalg.norm(emd1-emd2)

    @staticmethod
    def draw_text(image,text,origin):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2 , cv2.LINE_AA)