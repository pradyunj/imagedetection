# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 17: Trajectory Prediction (Velocity Vectors & Predictive Collision Check)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

trail = []
rect_zone = (200, 100, 240, 280)  # (x, y, w, h)
zone_pts = np.array([[200, 100], [440, 100], [440, 380], [200, 380]], np.int32).reshape((-1, 1, 2))

print("[SYSTEM] Predictive Trajectory Vector system active.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model.predict(source=frame, verbose=False)

    predictive_alert = False

    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf > 0.40:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            trail.append((cx, cy))
            if len(trail) > 30:
                trail.pop(0)

            # Task 1: Velocity Vector Calculation
            if len(trail) >= 5:
                px, py = trail[-5]
                vx, vy = cx - px, cy - py
                
                # Task 2: Project forward vector (30 frames forward)
                fx = int(cx + (vx * 6))
                fy = int(cy + (vy * 6))

                # Task 3: Predictive Collision Check with Danger Zone
                hit, _, _ = cv2.clipLine(rect_zone, (cx, cy), (fx, fy))
                if hit:
                    predictive_alert = True
                    cv2.line(frame, (cx, cy), (fx, fy), (0, 0, 255), 3)
                    cv2.putText(frame, "PREDICTED COLLISION!", (cx, cy - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.line(frame, (cx, cy), (fx, fy), (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            break

    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_pts], (0, 0, 255))
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    cv2.polylines(frame, [zone_pts], True, (0, 0, 255), 2)

    cv2.imshow("Pradyun's AI - Predictive Vector Trajectory", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 17 Complete! Predictive velocity vector engine operational.")
