
import pandas as pd
import cv2
from ultralytics import YOLO
import time

model = YOLO('yolo12n.pt')
names=model.names

cap = cv2.VideoCapture('video/bright.mp4') ##either place filename here, or the video feed url here.
prev_time = time.time()

data = []
frame = 0


while True:
    confidences = []
    person = 0
    frame +=1

    success, img = cap.read()

    if not success:
        break

    results = model(img, stream=True)
    for r in results:
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                cv2.putText(img,f"FPS: {fps:.1f}",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),2)
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = names[class_id]
                    
                    
                    
                    if class_name == 'person':
                        person +=1
                        detected = person > 0
                        
                        if detected:
                             confidences.append(box.conf[0].item())
                        else:
                             confidence = 0
                             




                   
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                       
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        
                        cv2.putText(img, f'{names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 1)
                        cv2.putText(img, f"People: {person}",(20, 120), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),2)
    
    cv2.imshow('TapoVision', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    df = pd.DataFrame({
    "Confidence": confidences
})
    average_confidence = df["Confidence"].mean()

    data.append({
    "Frame": frame,
    "Detected": person,
    "Confidence": average_confidence,
    "FPS": fps
})

df = pd.DataFrame(data)
average_fps = df["FPS"].mean()

average_confidence = df["Confidence"].mean()

detection_rate = (df["Detected"] == 1).mean() * 100


print(df)
print("Average FPS:", average_fps)
print("Average Confidence:", average_confidence)
print("Detection Rate:", detection_rate, "%")



cap.release()
cv2.destroyAllWindows()