import cv2
import dlib
import numpy as np


class Face_utils:
    @staticmethod
    def detect_face(face_path):
        detector = dlib.get_frontal_face_detector()
        img = dlib.load_rgb_image(face_path)
        dets = detector(img,1)
        boxes=[]
        for i, d in enumerate(dets):
            box = (d.left(),d.top(),d.right(),d.bottom())
            boxes.append(box)
        return boxes

    @staticmethod
    def return_face(image,box):
        left, top, right, bottom = box
        roi = image[left:right,top:bottom]
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
