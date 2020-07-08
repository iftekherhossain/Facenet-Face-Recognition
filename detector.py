from face_main import Face_utils
import cv2


f = Face_utils()

path= "D:\\Facenet-Face_recognition\\nab1.jpg"
image = cv2.imread(path)
boxes = f.detect_face(path)

box = boxes[0]

roi = f.return_face(image,box)
print(roi.shape)
cv2.imshow("img",roi)
cv2.waitKey(0)
cv2.desttroyAllWindows()