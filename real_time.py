import dlib
import cv2
from face_main import Face_utils

cap = cv2.VideoCapture(0)
f= Face_utils()

cascade_path = "D:\\Facenet-Face_recognition\\haarcascade_frontalface_default.xml"
while True:
    ret, frame = cap.read()
    boxes = f.detect_face_haar_cascade(cascade_path,frame)
    check_tuple = type(boxes) is tuple

    if len(boxes)>=1 and not check_tuple:
        box = boxes[0]
        x,y,w,h = box[0],box[1],box[2],box[3]
        tup_box = (x,y,w,h)
        print(tup_box)
        if w>120 and h >120:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face = f.return_face(frame,tup_box)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    else:
        continue
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows() 