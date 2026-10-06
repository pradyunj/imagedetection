# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 7: Motion Tracking (Drawing the Center Point & Trajectory Trail)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Phase 2: Memory Array for tracking history (max 30 points)
trail_points = []
MAX_TRAIL_LENGTH = 30

print("[SYSTEM] Tracking trail active. Move target around the frame. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, verbose=False)
    
    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf > 0.40:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            
            # Phase 1: Center Point Mathematics: (x_c, y_c)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Draw center point dot
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            
            # Append center coordinate to trail array
            trail_points.append((cx, cy))
            if len(trail_points) > MAX_TRAIL_LENGTH:
                trail_points.pop(0)
            break

    # Phase 3: Connect historical dots to render trajectory tail
    for i in range(1, len(trail_points)):
        cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 255, 0), 2)

    cv2.imshow("Pradyun's AI - Trajectory Trail Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 7 Complete! Trajectory trail active.")
