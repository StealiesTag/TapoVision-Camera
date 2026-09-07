import cv2
import cvzone
from ultralytics import YOLO

model = YOLO('yolo12n.pt')
names=model.names