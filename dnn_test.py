import cv2
import numpy as np

path_proto = 'D:\\Facenet-Face_recognition\\deploy.prototxt.txt'
path_model = 'D:\\Facenet-Face_recognition\\res10_300x300_ssd_iter_140000.caffemodel'
im_path = 'D:\\Facenet-Face_recognition\\nabil1.jpg'
img = cv2.imread(im_path)

net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
def detect_face_dnn(net,image,con=0.9):
    boxes = []
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    (h, w) = image.shape[:2]
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < con:
            continue
        box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
        print("hellllloooooooooo" ,box)
        x1_,y1_,x2_,y2_ = box[0],box[1], box[2], box[3]
        w_ = x2_ - x1_
        h_ = y2_ - y1_
        box = [x1_,y1_,w_,h_]
        boxes.append(box)
    return boxes

boxes = detect_face_dnn(net,img)
box = boxes[0]
print(type(box[0]))
cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 2)
cv2.imshow("image",img)
cv2.waitKey(0)