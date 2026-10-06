# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 11: Zone Defense (The Wind Turbine Airspace Danger Polygon)
# ==============================================================================

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Phase 1 & 3: Coordinate Mapping of Turbine Danger Zone
# [Min X: 200, Max X: 440, Min Y: 100, Max Y: 380]
zone_pts = np.array([[200, 100], [440, 100], [440, 380], [200, 380]], np.int32).reshape((-1, 1, 2))

print("[SYSTEM] Airspace Danger Zone mapped. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    # Phase 2: Render semi-transparent Danger Zone overlay
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_pts], (0, 0, 255))  # Red fill
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(frame, [zone_pts], True, (0, 0, 255), 2)

    cv2.putText(frame, "TURBINE DANGER AIRSPACE", (210, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow("Pradyun's AI - Airspace Danger Polygon", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 11 Complete! Danger zone geofencing rendered.")
