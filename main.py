import cv2
from ultralytics import YOLO

model = YOLO('yolo12n.pt')
names=model.names

cap = cv2.VideoCapture('video/test.mp4')

while True:
    success, img = cap.read()
    if not success:
        break

    cv2.imshow('TapoVision', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()