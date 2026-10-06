# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 12: Collision Trigger (The Turbine Brake Mechanism)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

zone_pts = np.array([[200, 100], [440, 100], [440, 380], [200, 380]], np.int32).reshape((-1, 1, 2))

print("[SYSTEM] Turbine Collision Trigger active. Move inside the zone to test.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model.predict(source=frame, verbose=False)

    alert_active = False
    zone_color = (0, 0, 255)  # Default Red

    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf > 0.40:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Phase 1: Spatial Intersection Math
            if 200 <= cx <= 440 and 100 <= cy <= 380:
                alert_active = True
                # Phase 3: Console System Output
                print("[CRITICAL] TURBINE BRAKE ENGAGED - TARGET IN AIRSPACE!")

            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break

    # Phase 2: Visual Alarm - Flash Yellow
    if alert_active:
        if int(time.time() * 6) % 2 == 0:
            zone_color = (0, 255, 255)  # Flashing Yellow

    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_pts], zone_color)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    cv2.polylines(frame, [zone_pts], True, zone_color, 3)

    if alert_active:
        cv2.putText(frame, "BRAKE ENGAGED!", (230, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("Pradyun's AI - Collision Trigger & Turbine Brake", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 12 Complete! Collision brake trigger operational.")
