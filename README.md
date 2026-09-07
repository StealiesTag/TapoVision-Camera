# TapoVision-Camera

TapoVision is a real-time computer vision system that uses a Tapo CCTV camera at my house and YOLO to detect and count people in a live video stream. Since the camera system is pretty inconsistent with detection and sometimes just records any sort of movement, and sometimes alerts my family for no reason.
The project processes the camera feed with OpenCV, identifies people using object detection, and displays their locations and real-time count. It also evaluates detection and counting performance under different conditions to measure how reliably the system performs.




## Conclusion

The experiment showed that lighting had a noticeable effect on TapoVision's human-detection performance. Under bright and dim lighting, the system achieved F1 scores of 94.86% and 94.66%, respectively. However, performance decreased substantially under dark lighting, where the F1 score dropped to 68.98%.

The decrease was primarily caused by lower recall. While precision remained at 100% across all three conditions, recall decreased from 90.22% under bright lighting to 52.65% under dark lighting. This indicates that the system generally produced reliable detections when it identified a person, but became significantly more likely to miss people in darker conditions.

Average confidence also decreased as lighting became darker, from 0.776 under bright lighting to 0.663 under dark lighting. In contrast, average FPS remained relatively consistent across all conditions at approximately 21 FPS, suggesting that lighting had little effect on the processing speed of the system.

### Limitations

This experiment was conducted using a limited number of test videos and lighting conditions, so the results may not represent performance in every real-world environment. The evaluation also focused on whether a person was present rather than whether the system accurately counted multiple people. Additionally, only the model's confidence and the visual lighting conditions were varied; other factors such as camera angle, distance, occlusion, and movement could also affect detection performance.

Overall, the results demonstrate that TapoVision can perform reliable real-time human detection under normal lighting, while significantly darker environments remain a challenge for the system.



If you also have a camera feed available, follow these steps to test your live feed:


## Live Tapo Camera Feed

TapoVision can process a live video stream from a compatible Tapo camera using RTSP.

### 1. Enable camera streaming

Open the **Tapo app** and configure the camera's camera-account/streaming settings. Create a dedicated camera username and password for RTSP access.

> Do not use or publish your Tapo account password. Keep your camera credentials private.

### 2. Find the camera's local IP address

Make sure your computer and Tapo camera are connected to the same local network.

Find the camera's local IP address through your router or the Tapo app.

### 3. Create the RTSP URL

The general RTSP format is:

```text
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1
```

For example:

```text
rtsp://myuser:mypassword@192.168.1.100:554/stream1
```

Some cameras also provide a lower-resolution stream through:

```text
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream2
```

Use the stream supported by your camera.

### 4. Test the stream

Before running TapoVision, test the RTSP URL using a media player such as VLC.

If the live camera feed displays correctly, the stream is ready to be used by OpenCV.

### 5. Connect the stream to TapoVision

In the Python script, replace the video file path:

```python
cap = cv2.VideoCapture("video/test4.mp4")
```

with your RTSP URL:

```python
cap = cv2.VideoCapture(
    "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"
)
```

The rest of the detection pipeline remains the same:

```text
Tapo Camera
     ↓
    RTSP
     ↓
   OpenCV
     ↓
    YOLO
     ↓
Person Detection
     ↓
Occupancy Count
```

### Security

Never commit your RTSP username or password to GitHub.

For a public repository, use environment variables or a local configuration file that is excluded through `.gitignore`.

For example:

```text
.env
```

should be included in `.gitignore` so your credentials are not uploaded to the repository.



Alright, thanks for viewing my project!
