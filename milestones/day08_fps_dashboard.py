# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 8: Performance Analytics (The Real-Time FPS Dashboard)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

print("[SYSTEM] FPS Performance Dashboard online. Press 'q' to quit.")

while True:
    # Phase 1: Track the clock start
    start_time = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 3: Optimization Hack - Downscale frame to boost inference throughput
    frame = cv2.resize(frame, (640, 480))

    results = model.predict(source=frame, verbose=False)
    
    # Phase 1: Track loop finish time and calculate latency
    end_time = time.perf_counter()
    loop_latency = end_time - start_time
    
    # Phase 2: Calculate FPS = 1 / latency
    fps = int(1.0 / loop_latency) if loop_latency > 0 else 0
    
    # Render FPS Performance HUD overlay
    cv2.rectangle(frame, (10, 10), (160, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Pradyun's AI - FPS Analytics HUD", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 8 Complete! FPS telemetry HUD verified.")
