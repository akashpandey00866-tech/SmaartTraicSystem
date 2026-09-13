import cv2
from ultralytics import YOLO

# YOLO model load karein
model = YOLO('yolov8n.pt')

# YouTube ki jagah direct high-quality sample video link
video_url = "https://media.githubusercontent.com/media/ultralytics/yolov5/master/data/images/highway.mp4"

print("Fetching video from cloud... Starting AI Detection!")

cap = cv2.VideoCapture(video_url)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # AI Detection
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow("AI Traffic Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()