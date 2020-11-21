import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf


class Face_utils:
    @staticmethod
    def detect_face(face_path):
        detector = dlib.get_frontal_face_detector()
        img = dlib.load_rgb_image(face_path)
        dets = detector(img, 1)
        boxes = []
        for i, d in enumerate(dets):
            (x, y, w, h) = face_utils.rect_to_bb(d)
            box = (x, y, w, h)
            boxes.append(box)
        return boxes

    @staticmethod
    def detect_face_haar_cascade(haar_cascade_path, image, sc=1.3, mn=5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(haar_cascade_path)

        faces = cascade.detectMultiScale(gray, scaleFactor=sc, minNeighbors=mn)
        if (len(faces) == 0):
            return None, None
        return faces

    @staticmethod
    def detect_face_dnn(net, image, con=0.9):
        boxes = []
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        (h, w) = image.shape[:2]
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < con:
                continue
            box = (detections[0, 0, i, 3:7] *
                   np.array([w, h, w, h])).astype("int")
            x1_, y1_, x2_, y2_ = box[0], box[1], box[2], box[3]
            w_ = x2_ - x1_
            h_ = y2_ - y1_
            box = [x1_, y1_, w_, h_]
            boxes.append(box)
        return boxes

    @staticmethod
    def detect_face_dlib(image):
        detector = dlib.get_frontal_face_detector()
        #img = dlib.load_rgb_image(face_path)
        dets = detector(image, 1)
        boxes = []
        for i, d in enumerate(dets):
            (x, y, w, h) = face_utils.rect_to_bb(d)
            box = (x, y, w, h)
            boxes.append(box)
        return boxes

    @staticmethod
    def detect_face_mtcnn(detector, image, con=0):
        faces = detector.detect_faces(image)
        boxes = []
        for face in faces:
            if face['confidence'] > con:
                boxes.append(face['box'])
        return boxes

    @staticmethod
    def return_face(image, box):
        x, y, w, h = box
        roi = image[y:y+h, x:x+w]
        roi_resize = cv2.resize(roi, (160, 160))
        return roi_resize

    @staticmethod
    def face_embedding(model, roi):
        roi = np.array(roi)
        face_pix = roi.astype('float32')
        mean, std = face_pix.mean(), face_pix.std()
        face_pix = (face_pix-mean) / std
        sample = np.expand_dims(face_pix, axis=0)
        emd = model.predict(sample)
        return emd[0]

    @staticmethod
    def compare_embeddings(emd1, emd2):
        return np.linalg.norm(emd1-emd2)

    @staticmethod
    def draw_text(image, text, origin):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
