# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 3: Color Manipulation & Side-by-Side Multi-View HUD
# ==============================================================================

import cv2
import numpy as np

print("[SYSTEM] Starting Day 3: Color Manipulation Pipeline...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not available. Exiting.")
    exit()

print("[SYSTEM] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for uniform display
    frame = cv2.resize(frame, (480, 360))

    # Phase 1: Convert BGR to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert single-channel grayscale back to 3-channel for stacking compatibility
    gray_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    cv2.putText(gray_3ch, "Grayscale (B&W)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Phase 2: Surgical Color Isolation - Pure Red Channel
    # Zero out Blue (index 0) and Green (index 1) channels
    red_feed = frame.copy()
    red_feed[:, :, 0] = 0  # Zero Blue
    red_feed[:, :, 1] = 0  # Zero Green
    cv2.putText(red_feed, "Isolated Red Channel", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Label original frame
    cv2.putText(frame, "Original Feed", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Phase 3: Multi-View Dashboard - Stack horizontally
    dashboard = np.hstack([frame, red_feed])

    cv2.imshow("Pradyun's AI - Side-by-Side Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 3 Complete! Multi-view dashboard closed.")
