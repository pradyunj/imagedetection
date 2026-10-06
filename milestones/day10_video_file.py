# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 10: The Field Test (Local Video File Asset Processing)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Phase 1 & 2: Local Video Asset or Webcam fallback
video_file = "birdtest.mp4"
if os.path.exists(video_file):
    print(f"[SYSTEM] Ingesting local video file: {video_file}")
    cap = cv2.VideoCapture(video_file)
else:
    print(f"[SYSTEM] '{video_file}' not found. Falling back to live webcam index 0.")
    cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Phase 3: Video files end - break cleanly on final frame
    if not ret:
        print("[SYSTEM] Video stream ended cleanly.")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(source=frame, verbose=False)

    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf > 0.40:
            cls_name = model.names[int(box.cls[0].item())]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, f"{cls_name}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    cv2.imshow("Pradyun's AI - Video File Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 10 Complete! Field video processing tested.")
