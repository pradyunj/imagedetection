# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 5: The Live Bounding Box (Week 1 Climax - Real-Time YOLO Inference)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
from ultralytics import YOLO

print("[SYSTEM] Initializing YOLOv8 nano model for live video stream...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not available. Exiting.")
    exit()

print("[SYSTEM] Live detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 1: Pass live frame to YOLO model
    results = model.predict(source=frame, verbose=False)
    
    # Phase 2 & 3: Draw bounding boxes and text overlays
    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf > 0.40:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            
            # Draw green bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with class name and confidence score
            label = f"{cls_name}: {conf:.0%}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Pradyun's AI - Live Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 5 Complete! Week 1 Climax achieved.")
